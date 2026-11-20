import torch
from scipy.spatial.transform import Rotation as R


def interpolate_depth(xyz, multi_scale_features, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_)
    return multi_scale_xyz


def project_depth_v2_batch(depth_img: torch.Tensor, pos: torch.Tensor, quat_wb: torch.Tensor, fov: torch.Tensor):
    """Back-project batched depth maps to world coordinates.

    Args:
        depth_img: (B, H, W) depth along the camera forward axis.
        pos: (B, 3) camera positions.
        quat_wb: (B, 4) body-to-world quaternions in scipy (x, y, z, w) order.
        fov: (B, 1) horizontal field of view in degrees.
    Returns:
        (B, H*W, 3) world points.
    """
    B, d_H, d_W = depth_img.shape
    device = depth_img.device
    fov_rad = torch.deg2rad(fov)
    fx = (d_W * 0.5) / torch.tan(fov_rad * 0.5)
    cx = d_W / 2.0
    cy = d_H / 2.0

    r_wb_list = []
    for i in range(B):
        r = R.from_quat(quat_wb[i].cpu().numpy()).as_matrix()
        r_wb_list.append(torch.from_numpy(r).float())
    r_wb = torch.stack(r_wb_list, dim=0).to(device)

    v_coords, u_coords = torch.meshgrid(
        torch.arange(d_H, device=device),
        torch.arange(d_W, device=device),
        indexing='ij'
    )
    d = depth_img.reshape(B, -1)
    v = v_coords.reshape(-1).unsqueeze(0).expand(B, -1)
    u = u_coords.reshape(-1).unsqueeze(0).expand(B, -1)

    x = d
    y = (u.float() - cx) * d / fx
    z = (v.float() - cy) * d / fx
    pointcloud = torch.stack([x, y, z], dim=1)
    pointcloud_world = torch.bmm(r_wb, pointcloud.float()) + pos.unsqueeze(2)
    return pointcloud_world.permute(0, 2, 1).float()


def create_points(depth_img, params, poses):
    """Return (B, V, H, W, 3) world points from depths, [quat_wb, fov] params, and positions."""
    (B, V, H, W) = depth_img.shape
    depth_imgs = depth_img.flatten(0, 1)
    poses = poses.flatten(0, 1)
    quat_wbs = params[:, :, :4].flatten(0, 1)
    fovs = params[:, :, 4:].flatten(0, 1)
    pointclouds = project_depth_v2_batch(depth_imgs, poses, quat_wbs, fovs)
    return pointclouds.reshape(B, V, H, W, 3)
