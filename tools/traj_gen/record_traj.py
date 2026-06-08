import numpy as np
import logging
import io
import airsim
from typing import List
import math
from scipy.spatial.transform import Rotation as R
import airsim
import numpy as np
import time
import tqdm
import cv2
import os
import json
from typing import List
import math
from scipy.spatial.transform import Rotation as R
from PIL import Image
import os
import sys
import json
import pickle
import yaml
import numpy as np
import threading
from pathlib import Path
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

sys.path.append('./')
from pcd_gen.run_airsim import AirSimRunner



class ImageSaver:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
    def save_image(self, img, save_path):
        def save_img():
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if isinstance(img, Image.Image):  # Fix: Image.Image.
                img.save(save_path)
            elif isinstance(img, np.ndarray):
                if save_path.endswith('.png') or save_path.endswith('.jpg'):
                    Image.fromarray(img).save(save_path)  # Fix: convert before saving the image.
                else:
                    np.save(save_path, img)  # Fix: pass the img argument.
        
        return self.executor.submit(save_img)

        
class AirsimTrajRecord:
    def __init__(self, airsim_port=30001,send_freq=10.0,enable_record_video=False):
        self.airsim_port = airsim_port
        self.send_freq = send_freq
        self.image_frame = 1
        
        # Initialize the AirSim connection.
        self._init_airsim_connection()
        self.img_saver = ImageSaver()
        # Ensure the output directory exists.
        if enable_record_video:
            while True:
                try:
                    self._client2 = airsim.MultirotorClient(port=self.airsim_port)
                    self._client2.confirmConnection()
                    print(f'AirSim client 2 connected successfully on port {self.airsim_port}!')
                    break
                except Exception as e:
                    print(f"连接 AirSim 失败: {e}")
                    time.sleep(2)

    def _init_airsim_connection(self):
        """Initialize the AirSim connection."""
        while True:
            try:
                self._client = airsim.MultirotorClient(port=self.airsim_port)
                self._client.confirmConnection()
                self._client.enableApiControl(True)
                self._client.armDisarm(True)
                print(f'AirSim connected successfully on port {self.airsim_port}!')
                break
            except Exception as e:
                print(f"连接 AirSim 失败: {e}")
                time.sleep(2)
    
    def _set_camera_pose(self, x, y, z, yaw, pitch, roll):
        """Set the camera pose."""
        # Stop motion.
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        
        # Convert Euler angles to a quaternion.
        x_val, y_val, z_val, w_val  = R.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
        # Set pose.
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
    
    def _capture_images(self, camera_names=["0"],capture_depth=False,):
        """Capture and save images."""
        img_dict = {}
        for camera_name in camera_names:
            req_list = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene)]
            if capture_depth:
                req_list.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True,False))
            camera_responses = self._client.simGetImages(req_list)
            camera_info = self._client.simGetCameraInfo(camera_name)
            metainfo = {
                "fov": camera_info.fov,
            }
            rgb_response = camera_responses[0]
            rgb_img_buffer = io.BytesIO(rgb_response.image_data_uint8)
            rgb_img = Image.open(rgb_img_buffer)
            rgb_img = np.array(rgb_img)
            
            if capture_depth:
                depth_response = camera_responses[1]
                depth_img = np.array(depth_response.image_data_float, dtype=np.float32).reshape(
                            depth_response.height, depth_response.width)
                # print(f'{camera_name} depth_img ({depth_response.height},{depth_response.width}):',depth_img.shape)
                metainfo['width'] =  depth_response.width
                metainfo['height'] =  depth_response.height
                quat_wb =  depth_response.camera_orientation
                world_pos = depth_response.camera_position
                metainfo['quat_wb'] = [quat_wb.x_val, quat_wb.y_val, quat_wb.z_val, quat_wb.w_val]
                metainfo['camera_pos'] = [world_pos.x_val, world_pos.y_val, world_pos.z_val]
                
            else:
                depth_img = None
                # camera_info = airsim_bridge._client.simGetCameraInfo(camera_name)
                # fov = np.deg2rad(camera_info.fov)
            img_dict[camera_name] = {
                'depth': depth_img if depth_img is not None else None,
                'rgb': rgb_img,
                'metainfo': metainfo,
            }
        return img_dict
    
    def start_recording(self,record_camera="uav_on_0", interval=0.2,temp_folder="temp/videos"):
        self.recording = True
        self.frame_count = 0
        self.interval = interval
        self.video_folder = temp_folder
        os.makedirs(temp_folder,exist_ok=True)
        for file in os.listdir(self.video_folder):
            if file.endswith('.png'):
                os.remove(os.path.join(self.video_folder, file))
        def capture_loop():
            while self.recording:
                try:
                    responses = self._client2.simGetImages([
                        airsim.ImageRequest(record_camera, airsim.ImageType.Scene)
                    ])
                    if responses:
                        img_buffer = io.BytesIO(responses[0].image_data_uint8)
                        img = Image.open(img_buffer)
                        frame_filename = os.path.join(self.video_folder, f"frame_{self.frame_count:04d}.png")
                        self.img_saver.save_image(img,frame_filename)
                        self.frame_count += 1
                except Exception as e:
                    print(f"录制错误: {e}")
                time.sleep(interval)
        # Run recording in a background thread.
        self.recording_thread = threading.Thread(target=capture_loop)
        self.recording_thread.start()
        
    def stop_recording(self,output_video="drone_flight.mp4", speed_multiplier=1.0):
        """Stop recording."""
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        print(f"录制完成，共保存 {self.frame_count} 帧")
        if self.frame_count > 0:
            fps = (1.0 / self.interval) * speed_multiplier
            self.create_video(output_video, fps)
        else:
            print("没有帧可以生成视频")
            
    def create_video(self, output_filename="drone_flight.mp4", fps=30):
        frame_files = [f for f in os.listdir(self.video_folder) if f.endswith('.png')]
        frame_files.sort()
        if not frame_files:
            print("没有找到帧文件")
            return
        # Read the first frame to get the frame size.
        first_frame = cv2.imread(os.path.join(self.video_folder, frame_files[0]))
        height, width, _ = first_frame.shape
        # Create the video writer.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        # Write all frames.
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.video_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)

        video_writer.release()
        duration = len(frame_files) / fps
        print(f"视频已保存：{output_filename} 参数: {width}x{height}, {fps:.1f}fps, {len(frame_files)}帧,{duration:.1f}秒")

    def record_traj(self, record_list: List[np.ndarray], camera_names=["0"],record_video=False,capture_depth=False):
        """
        Record trajectory images.
        Args:
            record_list: pose list, each item is a [x, y, z, yaw] numpy array
            camera_names: camera name list
            is_takeoff_landing: whether to add takeoff and landing actions
        """
        if len(record_list) < 1:
            print("记录列表为空，退出")
            return
        # print(f"Start recording trajectory with {len(record_list)} pose points")
        if record_video:
            assert hasattr(self,'_client2')
            self.start_recording()
        # Reset frame counter.
        # Process each pose point.
        camera_img_list = []
        for i, pose in enumerate(record_list):
            yaw = pose[3] if len(pose) > 3 else 0.0
            pos = pose[:3]
            # Apply scale ratio.
            # print(f"Processing pose point {i+1}/{len(record_list)}: pos={pos}, yaw={yaw:.2f}")
            # Set camera pose.
            self._set_camera_pose(pos[0], pos[1], pos[2],yaw, 0, 0)
            time.sleep(1.0 / self.send_freq)
            img_dict = self._capture_images(camera_names,capture_depth=capture_depth)
            camera_img_list.append(img_dict)
            
        # print(f"Trajectory recording completed, saved {len(record_list)} frames")
        if record_video:
            self.stop_recording()
        return camera_img_list
    
    def __del__(self):
        """Destructor."""
        if hasattr(self, '_client'):
            try:
                self._client.armDisarm(False)
                self._client.enableApiControl(False)
            except:
                pass
        print("AirsimTrajRecord 已释放")

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

def record_all_trajs(task_name,env_name,traj_list,airsim_port,output_folder,capture_depth=False):
    # traj_path = 'output/BrushifyUrban/0/0.json'
    airsim_record = AirsimTrajRecord(airsim_port=airsim_port,enable_record_video=False)
    async_img_saver = ImageSaver()
    
    base_json_folder = os.path.join(output_folder,'json',env_name)
    base_image_folder = os.path.join(output_folder,'images',env_name)
    # camera_names = ['uav_on_0','uav_on_1','uav_on_2','uav_on_3']
    camera_names = ['uav_on_0']
    for traj_row in tqdm.tqdm(traj_list,desc=f"[{task_name}-{env_name}]"):
        eps_id = traj_row['episode_id']
        pose_idx = traj_row['pose_idx']
        traj_data = json.load(open(traj_row['eps_path']))
        record_list = traj_data['record_list']
        # cur_image_folder = os.path.join(base_image_folder,")
        if len(record_list) < 1:
            continue
        camera_img_list = airsim_record.record_traj(record_list,camera_names=camera_names,capture_depth=capture_depth)
        camera_img_pathes = defaultdict(list)      
        for cur_i,cur_camera_img in enumerate(camera_img_list):
            for camera_name,img_dict in cur_camera_img.items():
                rgb_img = img_dict['rgb']
                
                rel_path = os.path.join(f"{eps_id}/{pose_idx}",f'{camera_name}',f'{cur_i:05d}.png' )
                img_filepath = os.path.join(base_image_folder,rel_path)
                async_img_saver.save_image(rgb_img,img_filepath)

                img_path_dict = {'rgb':rel_path,'metainfo':img_dict['metainfo']}
                if capture_depth:
                    d_img = img_dict['depth']
                    assert d_img is not None
                    d_rel_path = os.path.join(f"{eps_id}/{pose_idx}",f'{camera_name}_depth',f'd_{cur_i:05d}.npy' )
                    d_filepath = os.path.join(base_image_folder,d_rel_path)
                    os.makedirs(os.path.dirname(d_filepath),exist_ok=True)
                    np.save(d_filepath, d_img)
                    
                    img_path_dict['depth'] = d_rel_path
                    
                camera_img_pathes[camera_name].append(img_path_dict)

        traj_data['image_dict'] = camera_img_pathes
        json_save_path = os.path.join(base_json_folder,f"{eps_id}/{pose_idx}.json")
        os.makedirs(os.path.dirname(json_save_path),exist_ok=True)
        with open(json_save_path,'w') as f:
            json.dump(traj_data,f,indent=2)
        

def change_and_save_settings(air_port):
    base_settings = json.load(open('/home/zzz/code/UAV_ON/tools/configs/base_settings_512.json'))
    base_settings["ApiServerPort"] = air_port
    new_path = os.path.join(f"/home/zzz/code/UAV_ON/tools/configs/port_settings_512/settings_{air_port}.json")
    os.makedirs(os.path.dirname(new_path),exist_ok=True)
    with open(new_path,'w') as f:
        json.dump(base_settings,f)
    return new_path
        
def extract_already_save(env_folder):
    eps_folders = os.listdir(env_folder)
    traj_dict = {}
    for eps_name in eps_folders:
        eps_folder = os.path.join(env_folder,eps_name)
        for traj_name in os.listdir(eps_folder):
            traj_idx = os.path.splitext(traj_name)[0]  # Remove the extension.
            traj_dict[(eps_name,traj_idx)] = {
                'eps_path':os.path.join(eps_folder,traj_name),
                'episode_id':eps_name,
                'pose_idx':traj_idx,
            }
    return traj_dict
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_airsim_16', help="input env name")
    parser.add_argument('--base_folder', type=str, default='output', help="input env name")
    parser.add_argument('--output_folder', type=str, default='record_output', help="input env name")
    parser.add_argument('--capture_depth', type=bool, default=False, help="input env name")

    args = parser.parse_args()
    print(
        f' env:{args.env}\n base_folder: {args.base_folder}\n output_folder: {args.output_folder}\n capture_depth: {args.capture_depth}'
    )
    env_folder = os.path.join(args.base_folder,args.env)
    
    global_config_file = "configs/" + args.env + ".yaml"
    with open(global_config_file, 'r') as f:
        global_config = yaml.safe_load(f)

    eps_folders = os.listdir(env_folder)
    print('eps_folders examples:',[os.path.join(env_folder,eps) for eps in eps_folders[:3]])
    # for traj_path in os.listdir(eps_folders)
    traj_list = []
    for eps_name in eps_folders:
        eps_folder = os.path.join(env_folder,eps_name)
        for traj_name in os.listdir(eps_folder):
            traj_idx = os.path.splitext(traj_name)[0]  # Remove the extension.
            traj_list.append({
                'eps_path':os.path.join(eps_folder,traj_name),
                'episode_id':eps_name,
                'pose_idx':traj_idx,
            })
    
    planner_configs = global_config['thread_params']
    num_threads = len(planner_configs)
    print(len(eps_folders))
    
    base_json_folder = os.path.join(args.output_folder,'json',args.env)
    base_image_folder = os.path.join(args.output_folder,'images',args.env)
    os.makedirs(base_json_folder,exist_ok=True)
    os.makedirs(base_image_folder,exist_ok=True)
    
    already_json_dict = extract_already_save(base_json_folder)
    print(f'already saved: {len(already_json_dict)}/{len(traj_list)}')
    cur_total_traj_list = []
    for obj in traj_list:
        if (obj['episode_id'],obj['pose_idx']) in already_json_dict:
            continue
        cur_total_traj_list.append(obj)
    
    if len(cur_total_traj_list) > 0:
        air_runner = AirSimRunner(env_name=args.env)
        def run_air_sim(total_gpus = 4):
            air_configs = [
                (p_i%total_gpus,change_and_save_settings(planner_config['aim_port'])) for p_i,planner_config in enumerate(planner_configs)
            ]
            air_runner.run_multiple_envs(air_configs)
            
        air_sim_thread = threading.Thread(target=run_air_sim)
        air_sim_thread.start()
        print('waiting for running airsim')
        time.sleep(15)
        threads = []
        try:
            # traj_list = traj_list[:8]
            for thread_i,config in enumerate(planner_configs):
                airsim_port = config['aim_port']
                cur_traj_list = split_pos_list_by_rank(cur_total_traj_list,num_threads,thread_i)
                thread = threading.Thread(target=record_all_trajs, args=(f'Thread_{thread_i}',args.env,cur_traj_list,airsim_port,args.output_folder,args.capture_depth))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            air_runner.cleanup()
        air_runner.cleanup()
        
    print("All threads completed!")
