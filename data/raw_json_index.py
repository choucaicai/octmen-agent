"""Build a small byte-offset index for a top-level JSON episode array.

The index contains no images or depth arrays.  It lets the training loader seek
one episode at a time in a large ``train.json`` file, so every DDP rank can read
only its own episode range without loading the complete JSON document.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import orjson

# Keep this map identical to the map used by the historical TFDS builders.
ACTION_MAP = {
    0: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),
    6: np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),
    7: np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),
    8: np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    9: np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


def iter_json_object_bytes(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Iterator[Tuple[int, int, bytes]]:
    """Yield ``(start_byte, end_byte, object_bytes)`` for a JSON array.

    This is a small byte-level scanner.  It tracks quoted strings and escaped
    characters, so braces inside an instruction do not terminate an episode.
    Only the current episode is retained in memory.
    """
    depth = 0
    in_string = False
    escaped = False
    start = -1
    object_buf = bytearray()
    offset = 0
    saw_object = False

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for byte in chunk:
                if depth == 0:
                    if byte == 123:  # {
                        depth = 1
                        start = offset
                        object_buf = bytearray((byte,))
                        saw_object = True
                else:
                    object_buf.append(byte)
                    if in_string:
                        if escaped:
                            escaped = False
                        elif byte == 92:  # \\
                            escaped = True
                        elif byte == 34:  # "
                            in_string = False
                    else:
                        if byte == 34:  # "
                            in_string = True
                        elif byte == 123:  # {
                            depth += 1
                        elif byte == 125:  # }
                            depth -= 1
                            if depth == 0:
                                yield start, offset + 1, bytes(object_buf)
                                object_buf = bytearray()
                offset += 1

    if depth != 0 or in_string:
        raise ValueError(f"Malformed JSON array: unterminated object/string in {path}")
    if not saw_object:
        raise ValueError(f"No episode objects found in {path}")


def _as_int(value) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid action id: {value!r}") from exc


def aligned_action_ids(episode: Dict, camera_name: str) -> Tuple[list[int], int, int]:
    actions = episode.get("action") or episode.get("actions") or []
    image_dict = episode.get("image_dict") or {}
    frames = image_dict.get(camera_name) or []
    action_len, frame_len = len(actions), len(frames)
    n = min(action_len, frame_len)
    if n <= 0:
        return [], action_len, frame_len

    # Historical builders discarded the two terminal records and appended two
    # explicit stop actions.  This keeps token/action statistics compatible.
    raw_ids = [_as_int(x) for x in actions[:n]]
    if n >= 2:
        ids = raw_ids[:-2] + [0, 0]
    else:
        ids = raw_ids
    return ids, action_len, frame_len


def _stats(action_vectors: list[np.ndarray], num_trajectories: int, num_transitions: int) -> dict:
    if action_vectors:
        arr = np.asarray(action_vectors, dtype=np.float32)
    else:
        arr = np.zeros((1, 8), dtype=np.float32)
    # Avoid NaNs in a malformed source while keeping the index usable.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = arr.min(axis=0)
    high = arr.max(axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    q01 = np.percentile(arr, 1, axis=0)
    q99 = np.percentile(arr, 99, axis=0)
    action = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "max": high.tolist(),
        "min": low.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }
    zero = np.zeros(8, dtype=np.float32).tolist()
    return {
        "action": action,
        "proprio": {
            "mean": zero,
            "std": zero,
            "max": zero,
            "min": zero,
            "q01": zero,
            "q99": zero,
        },
        "num_transitions": int(num_transitions),
        "num_trajectories": int(num_trajectories),
    }


def index_matches_source(index: dict, source: Path, stat: os.stat_result, camera_name: str) -> bool:
    return (
        index.get("source") == str(source)
        and index.get("source_size") == stat.st_size
        and index.get("source_mtime_ns") == stat.st_mtime_ns
        and index.get("camera_name") == camera_name
    )


def build_index(source: Path, output: Path, camera_name: str = "uav_on_0", force: bool = False) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    stat = source.stat()
    if output.exists() and not force:
        try:
            old = orjson.loads(output.read_bytes())
            if index_matches_source(old, source, stat, camera_name):
                return old
        except Exception:
            pass

    episodes = []
    action_vectors: list[np.ndarray] = []
    total_transitions = 0
    mismatch_count = 0
    camera_mismatch_count = 0
    missing_target_count = 0
    malformed_count = 0

    for episode_id, (start, end, raw) in enumerate(iter_json_object_bytes(source)):
        try:
            episode = orjson.loads(raw)
            image_dict = episode.get("image_dict") or {}
            ids, action_len, frame_len = aligned_action_ids(episode, camera_name)
            camera_lengths = {str(k): len(v or []) for k, v in image_dict.items()}
            if len(set(camera_lengths.values())) > 1:
                camera_mismatch_count += 1
            if action_len != frame_len:
                mismatch_count += 1
            if "target" not in episode:
                missing_target_count += 1
            for action_id in ids:
                if action_id not in ACTION_MAP:
                    raise ValueError(f"unknown action id {action_id}")
                action_vectors.append(ACTION_MAP[action_id])
            total_transitions += len(ids)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "start": start,
                    "end": end,
                    "num_steps": len(ids),
                    "action_len": action_len,
                    "frame_len": frame_len,
                    "camera_lengths": camera_lengths,
                    "has_target": "target" in episode,
                    "instruction": bool(episode.get("gpt_instruction") or episode.get("instruction")),
                }
            )
        except Exception as exc:
            malformed_count += 1
            episodes.append(
                {
                    "episode_id": episode_id,
                    "start": start,
                    "end": end,
                    "num_steps": 0,
                    "action_len": 0,
                    "frame_len": 0,
                    "camera_lengths": {},
                    "has_target": False,
                    "instruction": False,
                    "error": str(exc),
                }
            )
        if (episode_id + 1) % 500 == 0:
            print(f"indexed {episode_id + 1} episodes", flush=True)

    result = {
        "format": "uav-on-json-byte-index-v1",
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "camera_name": camera_name,
        "episodes": episodes,
        "num_episodes": len(episodes),
        "num_transitions": total_transitions,
        "diagnostics": {
            "action_frame_mismatch": mismatch_count,
            "camera_length_mismatch": camera_mismatch_count,
            "missing_target": missing_target_count,
            "malformed": malformed_count,
        },
        "dataset_statistics": _stats(action_vectors, len(episodes), total_transitions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    os.replace(tmp, output)
    return result


def main() -> None:
    from utils.project_config import config_section, resolve_path

    train = config_section('train')
    data_dir = resolve_path(os.environ.get('TRAIN_DATA_DIR') or None)
    source = data_dir / 'train.json' if data_dir else resolve_path(train['raw_train_json'])
    output = resolve_path(train['raw_index_path'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=source, required=source is None, type=Path)
    parser.add_argument("--output", default=output, required=output is None, type=Path)
    parser.add_argument("--camera-name", default=train['raw_camera_name'])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_index(args.source, args.output, args.camera_name, args.force)
    print(
        f"index: {args.output}\n"
        f"episodes={result['num_episodes']} transitions={result['num_transitions']}\n"
        f"diagnostics={result['diagnostics']}"
    )


if __name__ == "__main__":
    main()
