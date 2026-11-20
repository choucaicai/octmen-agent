"""Factory for the independent hierarchical-memory training data path."""
from pathlib import Path
from .action_tokenizer import ActionTokenizer
from .raw_data_collector import RawDepthRLDSBatchTransform
from .raw_json_dataset import RawEpisodeDataset
from .memory_dataset import MemoryEpisodeDataset
from .memory_collator import MemoryWindowCollator


def get_memory_dataset_and_collator(*, train_json, val_json, media_root, index_path,
                                    image_transform, tokenizer, train, local_batch_size,
                                    tbptt_steps=2, seed=11, val_fraction=0.05,
                                    camera_name='uav_on_0'):
    source, source_index = Path(train_json), index_path
    statistics = None
    if not train and val_json is not None:
        source = Path(val_json)
        source_index = index_path if source.resolve() == Path(train_json).resolve() else None
        statistics = RawEpisodeDataset(train_json, index_path=index_path, camera_name=camera_name).dataset_statistics
    action_tokenizer = ActionTokenizer(tokenizer)
    transform = RawDepthRLDSBatchTransform(action_tokenizer, tokenizer, image_transform)
    dataset = MemoryEpisodeDataset(
        source, index_path=source_index, media_root=media_root, batch_transform=transform,
        train=train, seed=seed, val_fraction=val_fraction, camera_name=camera_name,
        split_by_episode=val_json is None,
        normalization_statistics=statistics, local_batch_size=local_batch_size, tbptt_steps=tbptt_steps,
    )
    return dataset, action_tokenizer, MemoryWindowCollator(tokenizer)
