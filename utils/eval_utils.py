"""AirSim observation and navigation helpers for UAV-ON evaluation."""
import io
import math
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from data.raw_json_index import ACTION_MAP
from .project_config import config_section, resolve_path

AIRSIM_ENV_ROOT = resolve_path(os.environ.get('AIRSIM_ENV_ROOT') or config_section('eval')['airsim_env_root'])
ENV_DICT = {
    name: resolve_path(path, relative_to=AIRSIM_ENV_ROOT)
    for name, path in config_section('airsim_scenes').items()
}


class AirSimObservations:
    def _set_camera_pose(self, x, y, z, yaw, pitch, roll):
        """Set the camera pose."""
        import airsim

        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        x_val, y_val, z_val, w_val = R.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
        target_pose = airsim.Pose()
        target_pose.position.x_val = float(x)
        target_pose.position.y_val = float(y)
        target_pose.position.z_val = float(z)

        target_pose.orientation.w_val = w_val
        target_pose.orientation.x_val = x_val
        target_pose.orientation.y_val = y_val
        target_pose.orientation.z_val = z_val

        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)

    def capture_rgbd(self, camera_name):
        import airsim

        rgb_response, depth_response = self._client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene),
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
        ])
        with Image.open(io.BytesIO(rgb_response.image_data_uint8)) as image:
            rgb = np.array(image)
        camera_info = self._client.simGetCameraInfo(camera_name)
        depth = np.array(depth_response.image_data_float, dtype=np.float32).reshape(
            depth_response.height, depth_response.width)
        quat_wb = depth_response.camera_orientation
        world_pos = depth_response.camera_position
        return {
            'depth': depth,
            'rgb': rgb,
            'metainfo': {
                'fov': camera_info.fov,
                'quat_wb': [quat_wb.x_val, quat_wb.y_val, quat_wb.z_val, quat_wb.w_val],
                'camera_pos': [world_pos.x_val, world_pos.y_val, world_pos.z_val],
            },
        }


def calculate_distance_uavon(position, target_position):
    obs_pos = np.array(position)
    coords = np.array(target_position)
    if coords.ndim == 2 and coords.shape[1] == 3:
        dists = np.linalg.norm(coords - obs_pos[None, :], axis=1)
        return float(dists.min())
    else:
        return float(np.linalg.norm(obs_pos - coords))


def convert_to_action_id(action):
    for action_id, value in ACTION_MAP.items():
        if np.array_equal(action, value):
            return action_id
    return 0




def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]


def to_eularian_angles(x_val = 0.0, y_val = 0.0, z_val = 0.0, w_val = 1.0):
    z = z_val
    y = y_val
    x = x_val
    w = w_val
    ysqr = y * y

    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)
