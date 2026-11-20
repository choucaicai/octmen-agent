"""Sparse episode memory with one default per block and fine-cell overrides."""
from dataclasses import dataclass, replace

import torch
from torch import nn


@dataclass
class MemoryState:
    keys: torch.Tensor
    features: torch.Tensor
    xyz_sum: torch.Tensor
    counts: torch.Tensor
    step_id: int

    @property
    def world_xyz(self):
        return self.xyz_sum / self.counts[:, None]

    def detach(self):
        return replace(self, features=self.features.detach())


class AdaptiveOctreeMemory(nn.Module):
    """Store prepared sparse voxel states by episode, without fusing observations."""

    def __init__(self):
        super().__init__()
        self.states = {}

    def get(self, episode_id):
        return self.states.get(str(episode_id))

    def save(self, episode_id, state):
        self.states[str(episode_id)] = state

    def reset(self, episode_id=None):
        if episode_id is None:
            self.states.clear()
        else:
            self.states.pop(str(episode_id), None)

    def detach(self):
        self.states = {key: value.detach() for key, value in self.states.items()}

    @staticmethod
    def lookup(state, queries):
        """Look up encoded keys, falling back to the stored block default."""
        defaults = block_default_keys(queries)
        groups, inverse = torch.unique(torch.cat((state.keys, queries, defaults)), dim=0, return_inverse=True)
        stored, fine, coarse = inverse.split((len(state.keys), len(queries), len(queries)))
        row_ids = stored.new_full((len(groups),), len(state.keys))
        row_ids[stored] = torch.arange(len(state.keys), device=stored.device)
        fine, coarse = row_ids[fine], row_ids[coarse]
        rows = torch.where(fine < len(state.keys), fine, coarse)
        values = torch.cat((state.features, state.features.new_zeros((1, state.features.shape[-1]))))
        return values[rows], rows < len(state.keys)

    @staticmethod
    def read(state):
        """Read each stored row once; a block default is never expanded into its fine cells."""
        return state.features, state.world_xyz, state.keys[:, 3] == 0


def block_default_keys(keys):
    return torch.cat((keys[:, :3], torch.zeros_like(keys[:, 3:])), dim=1)
