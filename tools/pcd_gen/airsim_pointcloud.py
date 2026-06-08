import airsim
import numpy as np  
import time
import open3d as o3d
from datetime import datetime
import threading
import glob
import yaml
import msgpackrpc
import shutil
from pathlib import Path
import os
from typing import List, Tuple, Optional
import sys
from scipy.spatial.transform import Rotation as R
sys.path.append(str(Path(__file__).parents[4]))
sys.path.append('./')
print(str(Path(__file__).parents[4]))
import json
import re
import argparse
import subprocess
import tqdm
from tools.sim.common import *
import logging
import io
from PIL import Image
logger = logging.getLogger(__name__)
from run_airsim import AirSimRunner

def pcd_to_ply(pcd_file_path, ply_file_path):
    # Read the PCD file.
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    
    # Check whether loading succeeded.
    if len(point_cloud.points) == 0:
        print("PCD文件加载失败或为空")
        return False
    
    print(f"成功加载PCD文件: {len(point_cloud.points)} 个点")

    # Get point coordinates.
    points = np.asarray(point_cloud.points)
    z_values = points[:, 2]  # Z coordinate (height).
    
    # Compute the height range.
    z_min, z_max = z_values.min(), z_values.max()
    print(f"高度范围: [{z_min:.2f}, {z_max:.2f}]")
    
    # Normalize height values to [0, 1].
    if z_max > z_min:
        normalized_heights = (z_values - z_min) / (z_max - z_min)
    else:
        normalized_heights = np.zeros_like(z_values)
    
    # Create a color map: blue -> green -> yellow -> red.
    colors = np.zeros((len(points), 3))
    
    for i, height in enumerate(normalized_heights):
        if height < 0.25:  # Low: blue to cyan.
            colors[i] = [0, 4*height, 1]
        elif height < 0.5:  # Mid-low: cyan to green.
            colors[i] = [0, 1, 1-4*(height-0.25)]
        elif height < 0.75:  # Mid-high: green to yellow.
            colors[i] = [4*(height-0.5), 1, 0]
        else:  # High: yellow to red.
            colors[i] = [1, 1-4*(height-0.75), 0]
    
    # Apply colors to the point cloud.
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY.
    success = o3d.io.write_point_cloud(ply_file_path, point_cloud)
    # Save as PLY.
    # success = o3d.io.write_point_cloud(ply_file_path, point_cloud)
    if success:
        print(f"成功转换并保存为: {ply_file_path}")
        print("颜色映射: 蓝色(低) -> 绿色 -> 黄色 -> 红色(高)")
        return True
    else:
        print("转换失败")
        return False
    
env_exec_path_dict = {
    "BrushifyUrban": {###
        'bash_name': 'BrushifyUrban',
        'exec_path': 'BrushifyUrban',
    },
    "CabinLake": {###
        'bash_name': 'CabinLake',
        'exec_path': 'CabinLake',
    },
    "CityPark": {###
        'bash_name': 'CityPark',
        'exec_path': 'CityPark',
    },
    "DownTown": {###
        'exec_path': 'DownTown',
        'bash_name': 'DownTown1',
    },
    "Neighborhood": {###
        'bash_name': 'Neighborhood',
        'exec_path': 'Neighborhood',
    },
    "Slum": {###
        'bash_name': 'slum1',
        'exec_path': 'Slum',
    },
    "UrbanJapan": {###
        'bash_name': 'UrbanJapan',
        'exec_path': 'UrbanJapan',
    },
    "Venice": {###
        'bash_name': 'vinice_new1',
        'exec_path': 'Venice',
    },
    "WesternTown": {###
        'bash_name': 'WesternTown1',
        'exec_path': 'WesternTown',
    },
    "WinterTown": {###
        'bash_name': 'WinterTown1',
        'exec_path': 'WinterTown',
    },
}

def get_next_sequence_number(save_dir, env_name):
    files = os.listdir(save_dir + '/' + env_name)
    pattern = re.compile(rf"{env_name}_(\d{{5}})_")
    max_seq_num = 0

    for file in files:
        match = pattern.search(file)
        if match:
            seq_num = int(match.group(1))
            if seq_num > max_seq_num:
                max_seq_num = seq_num

    return max_seq_num + 1
class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None
class AirsimBridge:  
    def __init__(self, env_name,airsim_port=30001):  
        self.env_name = env_name
        self._client = None
        self.global_point_cnt = 0
        self.airsim_port = airsim_port
        while True:
            try:
                self._client = airsim.MultirotorClient(port=self.airsim_port)
                self._client.confirmConnection()
                self._client.enableApiControl(True)
                self._client.armDisarm(True)
                print('connected successful!')
                break
            except Exception as e:
                logger.info(f"启动场景失败 {self.airsim_port}".format(e))
                time.sleep(3)

    def __del__(self): 
        print("end")
    
    def _connection_check(self):  
        if self._client.confirmConnection():  
            print('Airsim connected successfully')  
            self._client.enableApiControl(True)
            self._client.armDisarm(True)
        else:  
            print('Airsim is not connected')  
            exit()
  
    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(roll, pitch, yaw)
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)

    def get_images(self, camera_names, update_frequency=10):
        responses = []
        for camera_name in camera_names:
            simGetCameraInfo = self._client.simGetCameraInfo(camera_name)
            camera_responses = self._client.simGetImages([
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.Scene),
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.DepthPlanar, True),
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.Segmentation)
            ])
            responses.extend(camera_responses)

        time.sleep(1 / update_frequency)
        return responses

    def process_images(self, camera_names, save_dir, env_name):
        seq_num = get_next_sequence_number(save_dir, env_name)
        images = self.get_images(camera_names)
        print("images len", len(images))
        print("save images")
        
        for i, response in enumerate(images):
            seq_str = f"{seq_num:05d}"
            image_types = {0: 'color', 1: 'depth', 5: 'object_mask'}
            filename = f"{save_dir}/{env_name}/{env_name}_{seq_str}_{response.camera_name}_{image_types[response.image_type]}"
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        
            if i == 0:
                pose_info = {
                    "id": f"{env_name}_{seq_str}_{response.camera_name}",
                    "pos": {
                        "x": response.camera_position.x_val,
                        "y": response.camera_position.y_val,
                        "z": response.camera_position.z_val
                    },
                    "orient": {
                        "w": response.camera_orientation.w_val,
                        "x": response.camera_orientation.x_val,
                        "y": response.camera_orientation.y_val,
                        "z": response.camera_orientation.z_val
                    }
                }
                jsonl_filename = f"{save_dir}/{env_name}/{env_name}.jsonl"
                with open(jsonl_filename, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(pose_info) + '\n')

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
        self._client.simSetVehiclePose(new_pose, True)
        
    def get_depth_data(self,camera_name="pcd_cam_front") -> Tuple[np.ndarray, airsim.ImageResponse]:
        image_requests = [
            airsim.ImageRequest(
                camera_name, airsim.ImageType.DepthPlanar, True, False
            ),
        ]
        responses = self._client.simGetImages(image_requests)
        response = responses[0]
        depth_img = np.array(response.image_data_float, dtype=np.float32).reshape(
            response.height, response.width
        )
        # print('depth_img:',depth_img.shape)
        return depth_img, response
    def get_rgb_data(self,camera_name="pcd_cam_front") -> Tuple[np.ndarray, airsim.ImageResponse]:
        image_requests = [
            airsim.ImageRequest(
                camera_name, airsim.ImageType.Scene,
            ),
        ]
        responses = self._client.simGetImages(image_requests)
        response = responses[0]
        rgb_img_buffer = io.BytesIO(response.image_data_uint8)
        rgb_img = Image.open(rgb_img_buffer)
        rgb_img = np.array(rgb_img)
        return rgb_img

    def process_lidar_data(self, file_path):
        point_cloud = self.get_lidar_data()
        return point_cloud
            
    def project_depth(self, depth_img: np.ndarray, response: airsim.ImageResponse,fov=None,max_depth = 500):
        width = response.width
        height = response.height
        assert fov is not None
        fx = (width * 0.5) / math.tan(fov * 0.5)
        cx = width / 2
        cy = height / 2
        
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
                pixel_cords.append([v,u])
                pointcloud.append([x, y, z])
        r_wb = R.from_quat(quat_wb).as_matrix()
        pointcloud = np.array(pointcloud, dtype=np.float32).transpose()
        if pointcloud.size == 0:
            return None
        pointcloud = (r_wb @ pointcloud + world_pos).T
        pixel_cords = np.array(pixel_cords).astype(np.int32)
        return pointcloud,pixel_cords
    
        
def save_pointcloud_as_ply(points, filename,voxel_size = 0.25,colors=None):
    points = points.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # print('colors',type(colors),colors.shape,colors[0])
    if colors is not None:
        colors = colors[:,:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        #  o3d.utility.Vector3dVector(color.astype(np.float32) / 255.0)

    pcd = pcd.remove_duplicated_points()
    pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(filename, pcd)
                
def merge_all_points(folder,merge_colors=False):
    # Get all PLY files in the folder.
    ply_files = glob.glob(os.path.join(folder, "*.ply"))
    if not ply_files:
        print(f"在文件夹 {folder} 中未找到ply文件")
        return np.array([]).reshape(0, 3)
    all_points = []
    all_colors = []
    # Read each PLY file.
    for ply_file in tqdm.tqdm(ply_files):
        try:
            pcd = o3d.io.read_point_cloud(ply_file)
            if len(pcd.points) > 0:
                all_points.append(pcd.points)
                if merge_colors:
                    all_colors.append(pcd.colors)
                # print(f"Read {ply_file}: {len(points)} points")
            else:
                print(f"警告: {ply_file} 为空点云")
        except Exception as e:
            print(f"读取 {ply_file} 失败: {e}")
    
    if not all_points:
        print("所有ply文件都无法读取或为空")
        return np.array([]).reshape(0, 3)
    merged_points = np.vstack(all_points)
    if merge_colors:
        merged_colors = np.vstack(all_colors)
        return merged_points,merged_colors
    return merged_points,None
def handle_depth_collect(pos_list,env_name,airsim_port,temp_path,collect_rgb=False):
    airsim_bridge = AirsimBridge(env_name,airsim_port=airsim_port)
    # points_scene = None
    camera_name = "pcd_cam_front"
    for p_i,(x,y,z,yaw) in tqdm.tqdm(enumerate(pos_list),total=len(pos_list)):
        airsim_bridge.set_camera_pose_from_euler(
                    [x, y, z], [yaw, 0, 0]
                )
        depth_img, response = airsim_bridge.get_depth_data(camera_name=camera_name)
        
        camera_info = airsim_bridge._client.simGetCameraInfo(camera_name)
        fov = np.deg2rad(camera_info.fov)
        pointcloud,pixel_cords = airsim_bridge.project_depth(depth_img, response,fov=fov,max_depth=100)
        
        if collect_rgb:
            rgb_img = airsim_bridge.get_rgb_data(camera_name=camera_name)
            colors = rgb_img[pixel_cords[:, 0], pixel_cords[:, 1]]
            colors = colors.astype(np.float32) / 255.0
        else:
            colors = None
        if pointcloud is not None:
            ply_output_path = os.path.join(
                temp_path,
                f"{int(x)}_{int(y)}_{int(z)}_{yaw}.ply",
            )
            save_pointcloud_as_ply(pointcloud, ply_output_path,colors=colors)
        

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def split_pos_list_by_rank(pos_list, total, rank):
    if rank < 0 or rank >= total:
        raise ValueError(f"rank必须在0到{total-1}之间，当前rank为{rank}")
    if total <= 0:
        raise ValueError("total必须大于0")
    
    # Compute the base number of items assigned to each rank.
    base_size = len(pos_list) // total
    # Compute the start and end indices for the current rank.
    start_idx = rank * base_size
    if rank == total - 1:
        # Assign all remaining data to the last rank.
        end_idx = len(pos_list)
    else:
        end_idx = start_idx + base_size
    
    return pos_list[start_idx:end_idx]

def change_and_save_settings(air_port):
    base_settings = json.load(open('/home/zzz/code/UAV_ON/tools/configs/base_settings.json'))
    base_settings["ApiServerPort"] = air_port
    new_path = os.path.join(f"/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_{air_port}.json")
    with open(new_path,'w') as f:
        json.dump(base_settings,f)
    return new_path
    
def run_tasks(global_configs):
    planner_configs = global_configs['thread_params']
    print('planner_configs:\n',planner_configs)
    # [1050,1150,-1070,-965,-10,10]
    map_bound = global_configs['traj_map']['MapBound']
    dx, dy, dz =global_configs['traj_map']['MapDelta']
    #   MapBound:                   [-500, 1500, -1300, 500, -200, 200]     # Global voxel map range, within which trajectory generation is also performed.
    #   MapDelta:                   [200, 200, 50]  
    x_min, x_max, y_min, y_max, z_min, z_max = map_bound
    pos_list = []
    for z in np.arange(z_min, z_max, dz):
        yaw_mode = 1
        for y in np.arange(y_min, y_max, dy):
            yaw_mode = 0 if yaw_mode == 1 else 1
            for x in np.arange(x_min, x_max, dx):
                yaw = 0 if yaw_mode == 0 else np.deg2rad(180)
                pos_list.append([x,y,z,yaw])
                # print([x,y,z])

    num_threads = len(planner_configs)
    env_name = global_configs['datagen']['env']
    file_path = f"tools/pcd_gen_data/{env_name}_point_map_finegrain_v2/"
    os.makedirs(file_path,exist_ok=True)
    temp_path = os.path.join(file_path,'temp_points',)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path,exist_ok=True)
    
    air_runner = AirSimRunner(env_name=env_name)
    def run_air_sim(total_gpus = 4):
        air_configs = [
            (p_i%total_gpus,change_and_save_settings(planner_config['aim_port'])) for p_i,planner_config in enumerate(planner_configs)
        ]
        air_runner.run_multiple_envs(air_configs)
    air_sim_thread = threading.Thread(target=run_air_sim)
    air_sim_thread.start()
    print('waiting for running airsim')
    
    
    time.sleep(10)
    threads = []
    try:
        for thread_i,config in enumerate(planner_configs):
            airsim_port = config['aim_port']
            cur_pos_list = split_pos_list_by_rank(pos_list,num_threads,thread_i)
            thread = threading.Thread(target=handle_depth_collect, args=(cur_pos_list,env_name,airsim_port,temp_path))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        merge_points,merged_colors = merge_all_points(temp_path)
        save_pointcloud_as_ply(
                            merge_points, 
                            os.path.join(file_path,f"depth_final.ply",),
                            colors=merged_colors
                    )
    except KeyboardInterrupt:
        air_runner.cleanup()
    air_runner.cleanup()

def main():
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_airsim_16', help="input env name")
    args = parser.parse_args()
    config_file = "tools/configs/" + args.env + ".yaml"
    global_configs = load_config(config_file)
    run_tasks(global_configs)
    
if __name__ == '__main__':
    main()
