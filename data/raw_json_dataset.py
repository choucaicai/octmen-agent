"""Streaming episode dataset for the raw UAV-ON JSON format.

The source file is a top-level JSON array.  A byte-offset index is used so DDP
ranks can seek to disjoint episode ranges.  Only one episode is materialized
at a time.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import orjson
from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info

from utils.action_statistics import RAW_DATASET_NAME
from .raw_json_index import build_index, index_matches_source


class RawEpisodeDataset(IterableDataset):
    """Yield complete episodes one by one from a large JSON array.

    ``episode_range`` is an exclusive ``(start, end)`` range over the index.
    The range is further partitioned across distributed ranks and DataLoader
    workers.  The index itself is small and is safe to load into memory.
    """

    def __init__(
        self,
        source: Path | str,
        *,
        index_path: Path | str | None = None,
        camera_name: str = "uav_on_0",
        shuffle: bool = False,
        seed: int = 11,
        episode_range: Optional[Tuple[int, int]] = None,
        auto_build_index: bool = True,
    ) -> None:
        super().__init__()
        self.source = Path(source).expanduser().resolve()
        self.camera_name = camera_name
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        if index_path is None:
            index_path = self.source.with_suffix(self.source.suffix + ".oct_agent.index.json")
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.exists():
            if not auto_build_index:
                raise FileNotFoundError(f"Missing raw JSON index: {self.index_path}")
            print(f"[oct_agent] building index: {self.index_path}", flush=True)
            build_index(self.source, self.index_path, camera_name=self.camera_name)
        self.index = orjson.loads(self.index_path.read_bytes())
        if not index_matches_source(self.index, self.source, self.source.stat(), self.camera_name):
            if not auto_build_index:
                raise ValueError(f"Index does not match source file or camera {self.camera_name!r}: {self.index_path}")
            print(f"[oct_agent] rebuilding stale index: {self.index_path}", flush=True)
            self.index = build_index(self.source, self.index_path, camera_name=self.camera_name)
        self.episodes = self.index["episodes"]
        self._episode_range = episode_range or (0, len(self.episodes))
        self.dataset_statistics = self.index.get("dataset_statistics", {})
        self.dataset_statistics.setdefault("raw_source", str(self.source))
        self.dataset_statistics.setdefault("index_path", str(self.index_path))
        self.dataset_statistics.setdefault("camera_name", self.camera_name)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _consumer() -> Tuple[int, int]:
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        return rank * workers + worker_id, world * workers

    def _assigned_range(self) -> Tuple[int, int]:
        consumer, consumers = self._consumer()
        begin, end = self._episode_range
        count = max(0, end - begin)
        local_begin = begin + (count * consumer) // consumers
        local_end = begin + (count * (consumer + 1)) // consumers
        return local_begin, local_end

    def iter_episodes(self) -> Iterator[Dict[str, Any]]:
        begin, end = self._assigned_range()
        indices = list(range(begin, end))
        if self.shuffle and len(indices) > 1:
            consumer, _ = self._consumer()
            random.Random(self.seed + self.epoch * 1009 + consumer).shuffle(indices)

        # The file descriptor is shared only within this iterator and is never
        # held across workers.  Each worker/rank seeks only its assigned spans.
        with self.source.open("rb") as f:
            for index_id in indices:
                metadata = self.episodes[index_id]
                if metadata.get("num_steps", 0) <= 0 or metadata.get("error"):
                    continue
                f.seek(int(metadata["start"]))
                raw = f.read(int(metadata["end"]) - int(metadata["start"]))
                episode = orjson.loads(raw)
                episode["_source_index"] = index_id
                yield episode

    def local_partition(self) -> Dict[str, int]:
        begin, end = self._assigned_range()
        return {
            "episode_begin": begin,
            "episode_end": end,
            "num_episodes": end - begin,
            "num_steps": sum(int(x.get("num_steps", 0)) for x in self.episodes[begin:end]),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.iter_episodes()

    def __len__(self) -> int:
        begin, end = self._episode_range
        return int(sum(int(x.get("num_steps", 0)) for x in self.episodes[begin:end]))


class RawJSONVLAIterableDataset(IterableDataset):
    """Split indexed episodes and load their RGB-D frames and normalized actions."""

    def __init__(
        self,
        source: Path | str,
        *,
        index_path: Path | str | None,
        media_root: Path | str,
        batch_transform,
        train: bool,
        seed: int = 11,
        val_fraction: float = 0.05,
        camera_name: str = "uav_on_0",
        shuffle: Optional[bool] = None,
        auto_build_index: bool = True,
        repeat: Optional[bool] = None,
        split_by_episode: bool = True,
        normalization_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.media_root = Path(media_root).expanduser().resolve()
        self.batch_transform = batch_transform
        self.train = bool(train)
        self.repeat = self.train if repeat is None else bool(repeat)
        self.seed = int(seed)
        self.camera_name = camera_name
        self.shuffle = self.train if shuffle is None else bool(shuffle)
        val_fraction = float(val_fraction)
        if split_by_episode and not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        # Construct the small index-backed episode source once.  The train and
        # validation views share the index but use disjoint episode ranges.
        all_source = RawEpisodeDataset(
            source,
            index_path=index_path,
            camera_name=camera_name,
            shuffle=self.shuffle,
            seed=self.seed,
            auto_build_index=auto_build_index,
        )
        total = len(all_source.episodes)
        if split_by_episode and total < 2:
            raise ValueError("An episode train/validation split requires at least two episodes.")
        split_at = max(1, min(total - 1, int(round(total * (1.0 - val_fraction)))))
        if not split_by_episode:
            episode_range = (0, total)
        elif self.train:
            episode_range = (0, split_at)
        else:
            episode_range = (split_at, total)
        self.episodes = RawEpisodeDataset(
            source,
            index_path=all_source.index_path,
            camera_name=camera_name,
            shuffle=self.shuffle,
            seed=self.seed,
            episode_range=episode_range,
            auto_build_index=False,
        )
        stats = dict(self.episodes.dataset_statistics)
        if normalization_statistics is not None:
            stats["action"] = normalization_statistics["action"]
        self._normalization_statistics = stats
        stats["num_trajectories"] = int(episode_range[1] - episode_range[0])
        stats["num_transitions"] = len(self.episodes)
        stats["split"] = "train" if self.train else "val"
        stats["val_fraction"] = val_fraction if split_by_episode else 0.0
        # Checkpoints store action statistics under the dataset name used as `unnorm_key`.
        self.dataset_statistics = {RAW_DATASET_NAME: stats}

    def set_epoch(self, epoch: int) -> None:
        self.episodes.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.episodes)

    def _resolve_media(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.media_root / path
        return path

    def _load_rgb(self, value: Any) -> np.ndarray:
        path = self._resolve_media(value)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)

    def _load_depth(self, value: Any) -> np.ndarray:
        path = self._resolve_media(value)
        depth = np.load(path, allow_pickle=False)
        depth = np.asarray(depth)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
            if depth.ndim > 2:
                depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
        return np.asarray(depth, dtype=np.float32)

    @staticmethod
    def _normalise_action(action: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        action_stats = stats.get("action", {})
        low = np.asarray(action_stats.get("min", np.zeros(8)), dtype=np.float32)
        high = np.asarray(action_stats.get("max", np.ones(8)), dtype=np.float32)
        denom = high - low
        out = np.zeros_like(action, dtype=np.float32)
        valid = denom > 1e-8
        out[valid] = 2.0 * (action[valid] - low[valid]) / denom[valid] - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)
