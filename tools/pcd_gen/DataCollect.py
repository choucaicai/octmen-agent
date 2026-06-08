import os
import cv2
import json
import glob
import math
import time
import imageio
import numpy as np
import open3d as o3d
from tqdm import tqdm
import joblib as joblib
from pathlib import Path
import cosysairsim as airsim
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

# 27800 30500 975


def rotate(q_wb, pos_b):  # quat: wxzy
    pos_w = np.zeros_like(pos_b)
    if q_wb.ndim == 1:
        Rotation_wb = R.from_quat([q_wb[1], q_wb[2], q_wb[3], q_wb[0]])  # xyzw
        pos_w[:] = np.dot(Rotation_wb.as_matrix(), pos_b[:])
    else:
        for i in range(0, q_wb.shape[0]):
            Rotation_wb = R.from_quat(
                [q_wb[i, 1], q_wb[i, 2], q_wb[i, 3], q_wb[i, 0]]
            )  # xyzw
            pos_w[i, :] = np.dot(Rotation_wb.as_matrix(), pos_b[i, :])
    return pos_w


def transform(q_wb, tw, pos_b):
    pos_w = rotate(q_wb, pos_b)
    return pos_w + tw


class PointCloud_KDtree:
    def __init__(
        self,
        numpy_file_path=None,
        kdtree_file_path=None,
        pcd_file_path=None,
        voxel_size=None,
    ):
        self.numpy_file_path = numpy_file_path
        self.kdtree_file_path = kdtree_file_path
        self.pcd_file_path = pcd_file_path
        self.kdtree = None
        self.points = None
        self.pcd = None
        self.max_height_bound = 999
        self.load_file(voxel_size)

    def load_file(self, voxel_size=None, need_save=True):
        if self.numpy_file_path is not None:
            print("Loading numpy file:", self.numpy_file_path)
            self.points = np.load(self.numpy_file_path)
            print(self.points.shape)
        elif self.pcd_file_path is not None:
            print("Loading ply file:", self.pcd_file_path)
            pcd = o3d.io.read_point_cloud(self.pcd_file_path)
            if voxel_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            self.pcd = pcd
            self.points = np.asarray(pcd.points).astype(np.float32)
            if need_save:
                np.save("pcd_points.npy", self.points)
            print(self.points.shape)

        if self.kdtree_file_path is not None:
            print("Loading kdtree file:", self.kdtree_file_path)
            self.kdtree = joblib.load(self.kdtree_file_path)
        elif self.points is not None:
            print("Building kdtree")
            self.kdtree = cKDTree(self.points)
            if need_save:
                joblib.dump(self.kdtree, "big_kdtree.pkl")

        self.max_height_bound = np.max(self.points[:, 2])
        print("max_bound:", np.max(self.points, axis=0))
        print("min_bound:", np.min(self.points, axis=0))

    def check_collision(self, query_points, radius=1.5, frame_id="ros"):
        if frame_id == "ros":
            distances, _ = self.kdtree.query(
                query_points, k=1, distance_upper_bound=radius
            )
            collision = distances <= radius
        elif frame_id == "airsim":
            query_points = query_points.copy()
            query_points[1] *= -1
            query_points[2] *= -1
            distances, _ = self.kdtree.query(
                query_points, k=1, distance_upper_bound=radius
            )
            collision = distances <= radius
        return collision


class AirsimDataCollector:
    def __init__(
        self,
        camera_name="front_center",
        x_range=(-30, 30),
        y_range=(-30, 30),
        grid_step=5,
        save_path="./Datasets",
        collect_type="pcd",
        pointcloud_kdtree: PointCloud_KDtree = None,
    ):
        self.client = airsim.ComputerVisionClient()
        self.client.confirmConnection()
        self.camera_name = camera_name
        self.hfov = np.deg2rad(self.client.simGetCameraInfo(self.camera_name).fov)
        self.x_range = x_range
        self.y_range = y_range
        self.grid_step = grid_step
        self.save_path = save_path
        self.collect_type = collect_type
        self.pointcloud_kdtree = pointcloud_kdtree
        self.random_gen = np.random.default_rng()
        self.imgs_data = None  # Store image data for depth collection

    def get_image_data(self) -> Tuple[np.ndarray, airsim.ImageResponse]:
        image_requests = [
            airsim.ImageRequest(
                self.camera_name, airsim.ImageType.DepthPlanar, True, False
            ),
        ]
        responses = self.client.simGetImages(image_requests)
        response = responses[0]
        depth_img = np.array(response.image_data_float, dtype=np.float32).reshape(
            response.height, response.width
        )
        return depth_img, response

    def project_depth(self, depth_img: np.ndarray, response: airsim.ImageResponse):
        max_depth = 40
        width = response.width
        height = response.height
        fx = (width * 0.5) / math.tan(self.hfov * 0.5)
        cx = width / 2
        cy = height / 2
        pointcloud = []
        quat_wb = response.camera_orientation
        quat_wb = [quat_wb.x_val, quat_wb.y_val, quat_wb.z_val, quat_wb.w_val]
        world_pos = response.camera_position
        world_pos = np.array([[world_pos.x_val, world_pos.y_val, world_pos.z_val]]).T

        for v in range(height):
            for u in range(width):
                d = depth_img[v, u]
                if math.isnan(d) or d <= 0.001 or d > max_depth:
                    continue
                x = d
                y = (u - cx) * d / fx
                z = (v - cy) * d / fx
                pointcloud.append([x, y, z])
        r_wb = R.from_quat(quat_wb).as_matrix()
        pointcloud = np.array(pointcloud, dtype=np.float32).transpose()
        if pointcloud.size == 0:
            return None
        pointcloud = (r_wb @ pointcloud + world_pos).T
        return pointcloud

    def set_camera_pose_from_euler(
        self, pos: List[float], euler_angles: List[float]
    ):  # euler_angles: yaw pitch roll
        new_pose = airsim.Pose()
        new_pose.position.x_val = float(pos[0])
        new_pose.position.y_val = float(pos[1])
        new_pose.position.z_val = float(pos[2])

        x, y, z, w = R.from_euler("ZYX", euler_angles).as_quat()  # convert to xyzw

        new_pose.orientation.w_val = w
        new_pose.orientation.x_val = x
        new_pose.orientation.y_val = y
        new_pose.orientation.z_val = z

        self.client.simSetVehiclePose(new_pose, True)

    def set_camera_pose_from_quat(self, pos: List[float], quat):  # quat: wxzy
        new_pose = airsim.Pose()
        new_pose.position.x_val = float(pos[0])
        new_pose.position.y_val = float(pos[1])
        new_pose.position.z_val = float(pos[2])

        new_pose.orientation.w_val = quat[0]
        new_pose.orientation.x_val = quat[1]
        new_pose.orientation.y_val = quat[2]
        new_pose.orientation.z_val = quat[3]

        self.client.simSetVehiclePose(new_pose, True)

    def sample_random_state(self):
        x_length = self.x_range[1] - self.x_range[0]
        y_length = self.y_range[1] - self.y_range[0]
        center = np.array([0, 0, -1])
        scale = np.array([x_length, y_length, 1.5])
        roll_var = 0.01  # 5.73 degree
        pitch_var = 0.01 # 5.73 degree

        quad_state = {}
        uniform = self.random_gen.uniform
        normal = self.random_gen.normal
        while True:
            pos = 0.5 * scale * uniform(-1.0, 1.0, size=3) + center
            if not self.pointcloud_kdtree.check_collision(
                pos, radius=0.5, frame_id="airsim"
            ):
                quad_state["pos"] = pos
                break
        roll = normal() * np.sqrt(roll_var)
        pitch = normal() * np.sqrt(pitch_var)
        yaw = np.pi * uniform(-1.0, 1.0)
        quat = R.from_euler("ZYX", [yaw, pitch, roll]).as_quat()  # quat: xyzw
        quat = [quat[3], quat[0], quat[1], quat[2]]  # convert to wxzy
        quad_state["quat"] = quat
        return quad_state

    def save_pointcloud_as_ply(self, points, filename):
        points = points.astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=False)

    def collect_pointcloud(self, z_height):
        x_vals = np.linspace(
            self.x_range[0],
            self.x_range[1],
            int((self.x_range[1] - self.x_range[0]) / self.grid_step) + 1,
        )
        y_vals = np.linspace(
            self.y_range[0],
            self.y_range[1],
            int((self.y_range[1] - self.y_range[0]) / self.grid_step) + 1,
        )
        yaw_mode = 1
        for y_coord in tqdm(y_vals, desc="Y-axis"):
            yaw_mode = 0 if yaw_mode == 1 else 1
            for x_coord in tqdm(x_vals, desc=f"y={y_coord:.2f}", leave=False):
                yaw = 0 if yaw_mode == 0 else np.deg2rad(180)
                self.set_camera_pose_from_euler(
                    [x_coord, y_coord, z_height], [yaw, 0, 0]
                )
                depth_img, response = self.get_image_data()
                pointcloud = self.project_depth(depth_img, response)
                if pointcloud is not None:
                    output_path = os.path.join(
                        self.save_path,
                        f"{int(x_coord)}_{int(y_coord)}_{int(z_height)}_{yaw_mode}.ply",
                    )
                    self.save_pointcloud_as_ply(pointcloud, output_path)

    def collect_depth_images(self, index, max_depth=20):
        quat_state = self.sample_random_state()
        self.set_camera_pose_from_quat(quat_state["pos"], quat_state["quat"])
        # time.sleep(1)
        depth_img, response = self.get_image_data()
        depth_img = np.minimum(depth_img, max_depth)  # Ensure no negative values
        depth_img = depth_img / 20
        depth_img[np.isnan(depth_img)] = 1.0
        
        output_path = os.path.join(self.save_path, f"{index}.tif")
        # imageio.imwrite(output_path, depth_img)
        cv2.imwrite(output_path, depth_img)
        pos = response.camera_position
        quat = response.camera_orientation
        img_info = np.array([pos.x_val, pos.y_val, pos.z_val, quat.w_val, quat.x_val, quat.y_val, quat.z_val])
        self.imgs_data[index,:] = img_info

    def main_process(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.collect_type == "pcd":
            z_heights = [-0.5, -3.0]
            for z_height in z_heights:
                self.collect_pointcloud(z_height)
        elif self.collect_type == "depth":
            total_images = 10000
            self.imgs_data = np.zeros((total_images, 7), dtype=np.float32)  # Store position and quaternion
            for index in tqdm(range(total_images), desc="Collecting depth images"):
                self.collect_depth_images(index)
            np.save(
                os.path.join(self.save_path, "imgs_info.npy"), self.imgs_data
            )


def main():
    # kdtree = PointCloud_KDtree("pcd_points.npy", "big_kdtree.pkl", "world_ros_refine.ply")
    kdtree = PointCloud_KDtree(None, None, r"Datasets\4\world_ros.ply_0.1.ply")
    # kdtree = None
    collector = AirsimDataCollector(
        save_path="Datasets/4/0", collect_type="depth", pointcloud_kdtree=kdtree
    )
    collector.main_process()


if __name__ == "__main__":
    main()
