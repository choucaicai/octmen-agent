import math

import torch
import torch.nn as nn

from .multimodal_encoder.spatial_aware_module import SpatialAwareModule
from .multimodal_encoder.unproject import create_points, interpolate_depth

COARSE_SLOT = 0


def cartesian_to_spherical(feat_xyz, cur_pose):
    """Return relative xyz, radius, azimuth, and polar angle with respect to `cur_pose`."""
    N, V, H, W, _ = feat_xyz.shape
    if cur_pose.dim() == 1:
        cur_pose = cur_pose.unsqueeze(0)
    cur_pose = cur_pose.view(N, 1, 1, 1, 3)
    rel_xyz = feat_xyz - cur_pose
    x, y, z = rel_xyz[..., 0], rel_xyz[..., 1], rel_xyz[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.atan2(y, x)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    return torch.stack([x, y, z, r, phi, theta], dim=-1)


def encode_fine_keys(fine_xyz, width):
    """Map fine-cell indices [N, 3] to memory keys (block_x, block_y, block_z, slot) [N, 4].

    A block holds `width` fine cells per axis. Shifting by `width // 2` centres blocks on a fine
    cell when `width` is odd, so block b covers cells b*width - width//2 to b*width + width//2.
    COARSE_SLOT is the block default; fine cells use slots 1 to width**3.
    """
    shifted = fine_xyz + width // 2
    block_xyz = torch.div(shifted, width, rounding_mode='floor')
    offset_x, offset_y, offset_z = shifted.remainder(width).unbind(1)
    slot = 1 + offset_x * width ** 2 + offset_y * width + offset_z
    return torch.cat((block_xyz, slot[:, None]), dim=1)


class RGBDSceneTower(nn.Module):
    def __init__(self, latent_dim=1024, voxel_size=0.2, delay_load=False):
        super().__init__()
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('memory_voxel_size must be finite and positive')
        self.fine_voxel_size = voxel_size
        self.coarse_voxel_size = 25
        self.block_width = round(self.coarse_voxel_size / self.fine_voxel_size)
        assert math.isclose(self.block_width * self.fine_voxel_size, self.coarse_voxel_size), (
            f'coarse_voxel_size={self.coarse_voxel_size} must be an integer multiple of '
            f'memory_voxel_size={voxel_size}'
        )
        self.latent_dim = latent_dim
        self.pe_dim = 6
        self.fine_distance = 50
        self.max_distance = 500
        if not delay_load:
            self.load_model()

    def load_model(self):
        self.video_tower = SpatialAwareModule(dim=self.pe_dim, latent_dim=self.latent_dim)

    def encode_features_with_xyz(self, features_xyz):
        """Add the learned position embedding to features stored as [L, C + pe_dim]."""
        features, feature_xyz = torch.split(features_xyz, [self.latent_dim, self.pe_dim], dim=1)
        return features + self.video_tower.encode_pe(feature_xyz)

    def unproject_current(self, features, depths, params, poses):
        """Place each patch of the current frame in world coordinates.

        Returns patch features [B, P, C], world xyz [B, P, 3], a mask of finite points with positive
        depth within `max_distance`, and a mask of points within `fine_distance`.
        """
        batch, views, channels, height, width = features.shape
        if views != 1:
            raise ValueError('Memory accepts one new frame per environment step')
        with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
            world = create_points(depths.float(), params.float(), poses.float())
            patch_xyz = interpolate_depth(world, [features.flatten(0, 1)], method='nearest')[0].reshape(batch, -1, 3)
            patch_depth = torch.nn.functional.interpolate(
                depths.flatten(0, 1).unsqueeze(1).float(), size=(height, width), mode='nearest',
            ).reshape(batch, -1)
            camera_xyz = poses[:, -1:, :].float()
            distance = torch.linalg.vector_norm(patch_xyz - camera_xyz, dim=-1)
            valid = torch.isfinite(patch_xyz).all(-1) & torch.isfinite(patch_depth)
            valid = valid & (patch_depth > 0) & (distance <= self.max_distance)
            fine_mask = distance <= self.fine_distance
        patch_features = features.permute(0, 1, 3, 4, 2).reshape(batch, -1, channels)
        return patch_features, patch_xyz, valid, fine_mask

    def pool_current_voxels(self, features, world_xyz, fine_mask):
        """Average current-frame patches that share a memory key.

        Near patches (`fine_mask`) keep their fine-cell slot and far patches use their block's
        COARSE_SLOT, so one `unique` pools both levels without mixing them. Returns keys [K, 4],
        mean features [K, C], world xyz sums [K, 3], and patch counts [K].
        """
        world_xyz = world_xyz.float().detach()
        fine_xyz = torch.floor(world_xyz / self.fine_voxel_size + 0.5).long()
        keys = encode_fine_keys(fine_xyz, self.block_width)
        keys[:, 3].masked_fill_(~fine_mask, COARSE_SLOT)
        keys, patch_to_key, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
        feature_sum = world_xyz.new_zeros((len(keys), features.shape[-1])).index_add(0, patch_to_key, features.float())
        xyz_sum = world_xyz.new_zeros((len(keys), 3)).index_add(0, patch_to_key, world_xyz)
        mean_features = (feature_sum / counts[:, None]).to(features.dtype)
        return keys, mean_features, xyz_sum, counts
