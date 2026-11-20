# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Learned positional encoding for memory voxels.
"""
import torch.nn as nn


class PositionEmbeddingLearnedMLP(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super(PositionEmbeddingLearnedMLP, self).__init__()
        self.dim = dim
        self.position_embedding_head = nn.Sequential(
            nn.Linear(dim, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        shape_len = len(xyz.shape)
        if shape_len == 5:
            B, V, H, W, _ = xyz.shape
            xyz = xyz.flatten(1, 3)
        elif shape_len == 3 or shape_len == 2:
            xyz = xyz
        else:
            raise ValueError("xyz should be 2, 3 or 5 dimensional")
        position_embedding = self.position_embedding_head(xyz)
        if shape_len == 5:
            return position_embedding.reshape(B, V, H, W, -1)
        else:
            return position_embedding
