#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import pickle
import yaml
import numpy as np
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
# Import the compiled traj_gen_py library.
# import traj_gen_py
import open3d as o3d
import argparse
from path_search import VoxelMap,PathSearch
import numpy as np
import logging
import io
import math
from PIL import Image
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
# import airsim
# Create a quaternion object.
# quat = airsim.Quaternionr(0, 0, 0.7821841137294746, -0.6230473595395749)
# Convert to Euler angles.
def to_eularian_angles(x_val = 0.0, y_val = 0.0, z_val = 0.0, w_val = 1.0):
    z = z_val
    y = y_val
    x = x_val
    w = w_val
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return (pitch, roll, yaw)

def get_crop_box(points_array, map_bound, margin):
    """
    Compute a crop box while respecting global map bounds.
    Args:
        points_array: point-cloud array with shape (N, 3)
        map_bound: global map bounds [x_min, x_max, y_min, y_max, z_min, z_max]
        margin: margin; the final box size is 2 * margin
    
    Returns:
        bbox_min, bbox_max: minimum and maximum crop-box coordinates
    """
    # 1. Compute the point-cloud center.
    center_point = np.mean(points_array, axis=0)
    
    # 2. Global bounds.
    global_min = np.array([map_bound[0], map_bound[2], map_bound[4]])
    global_max = np.array([map_bound[1], map_bound[3], map_bound[5]])
    
    # 3. Compute bbox_min while keeping the full 2 * margin box inside global bounds.
    bbox_min = np.maximum(center_point - margin, global_min)      # Keep it above the global minimum.
    bbox_min = np.minimum(bbox_min, global_max - 2 * margin)      # Keep the right boundary inside.
    # 4. bbox_max = bbox_min + 2*margin
    bbox_max = bbox_min + 2 * margin
    
    # print(f'Center point: {center_point}')
    # print(f'BBox min: {bbox_min}')
    # print(f'BBox max: {bbox_max}')
    # print(f'BBox size: {bbox_max - bbox_min}, Margin: {margin}')
    
    return bbox_min, bbox_max

def crop_pointcloud_with_box(point_cloud, bbox_min,bbox_max,save_path=None):
    """
    Args:
        points_array: numpy array, shape (N, 3) - input point coordinates
        point_cloud: o3d.geometry.PointCloud - original point cloud
        margin: float - bounding-box margin, default 200
        save_path: str - save path; no file is saved when None
        visualize: bool - whether to visualize, default True
    Returns:
        cropped_point_cloud: o3d.geometry.PointCloud - cropped point cloud
        bbox: o3d.geometry.AxisAlignedBoundingBox - bounding box
        center_point: numpy array - computed center point
    """
    # 3. Create the bounding box.
    st_time = time.time()
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bbox_min,
        max_bound=bbox_max
    )
    bbox.color = [1, 0, 0]  # Red bounding box.
    # 4. Crop the point cloud.
    cropped_point_cloud = point_cloud.crop(bbox)
    # print(f'Original point cloud: {len(point_cloud.points)} points')
    print(f'Cropped point cloud spend {time.time() - st_time}: {len(point_cloud.points)} -> {len(cropped_point_cloud.points)} points')
    # 5. Save the cropped point cloud if a save path is provided.
    if save_path is not None:
        try:
            # Ensure the directory exists.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the point cloud.
            success = o3d.io.write_point_cloud(save_path, cropped_point_cloud)
            if success:
                print(f'Cropped point cloud saved to: {save_path}')
            else:
                print(f'Failed to save point cloud to: {save_path}')
        except Exception as e:
            print(f'Error saving point cloud: {e}')
            
    return cropped_point_cloud

def load_pcd_file(pcd_path: str) -> List[List[float]]:
    """Load a PCD file with Open3D."""
    st_time = time.time()
    # Load the PCD file with Open3D.
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        print(f"Warning: PCD file is empty or could not be loaded: {pcd_path}")
        return []
    
    logger.info(f"Successfully loaded {len(pcd.points)} points from {pcd_path}, spend: {time.time() - st_time}")
    return pcd
    


class PlannerConfig:
    """Planner configuration."""
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict.get("name", "")
        self.nums = config_dict.get("nums", 0)
        self.min_dis = config_dict.get("min_dis", 0.0)
        self.max_dis = config_dict.get("max_dis", 0.0)
        self.height_min = config_dict.get("height_min", 0.0)
        self.height_max = config_dict.get("height_max", 0.0)
        self.aim_port = config_dict.get("aim_port", 0)
        self.listen_port = config_dict.get("listen_port", 0)
        self.sim_ip = config_dict.get("sim_ip", "")
        self.aimlandmark_nums = config_dict.get("aimlandmark_nums", 0)
        self.add_takeoff_land = config_dict.get("add_takeoff_land", False)
        self.with_turn = config_dict.get("with_turn", False)


class ONPlanObject:
    """Planning object."""
    def __init__(self,episode_id,pose_idx,start_position,start_quaternion,target_position):
        self.episode_id: str = episode_id
        self.pose_idx = pose_idx
        self.start_position: List[float] = start_position
        self.start_quaternion: List[float] = start_quaternion
        self.target_position: List[float]= target_position

class PathPlannerNode:
    """Path planner node."""
    
    def __init__(self, task_name: str, env_name: str):
        self.task_name = task_name
        self.env_name = env_name
        self.pr_dir = get_project_directory()
        
        yaml_path = os.path.join(self.pr_dir, "configs", f"{env_name}.yaml")
        print(f"[{self.task_name}] Loading configuration from: {yaml_path}")
        
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        traj_map = yaml_data.get("traj_map", {})
        self.dilate_radius = traj_map.get("DilateRadius", 0.0)
        self.voxel_width = traj_map.get("VoxelWidth", 0.2)
        map_bound_list = traj_map.get("MapBound", [-50, 50, -50, 50, 0, 30])
        self.global_map_bound = list(map_bound_list)
        self.env_scale_ratio = traj_map.get("pcd_scale_ratio", 1.0)
        self.temp_dir = traj_map.get("temp_dir", 'temp')
        os.makedirs(self.temp_dir,exist_ok=True)
        
        print(f"[{self.task_name}] Configuration loaded successfully:")
        print(f"[{self.task_name}]   - Dilate Radius: {self.dilate_radius}")
        print(f"[{self.task_name}]   - Voxel Width: {self.voxel_width}")
        print(f"[{self.task_name}]   - Map Bounds: {self.global_map_bound}")
        
        self.path_searcher = None
        self.global_voxelmap = None

        self.current_map_name = env_name
        print(f"Starting PathPlanner initialization for task: {task_name}")
        # pcd_path = os.path.join(self.pr_dir, "scene_data", "pcd_map", f"{self.current_map_name}.pcd")
        ply_path = os.path.join(self.pr_dir, "scene_data", "pcd_map", f"{self.current_map_name}.ply")
        # print(f"[{self.task_name}] PCD path: {pcd_path}")
        print(f"[{self.task_name}] PLY path: {ply_path}")
        self.point_cloud = load_pcd_file(ply_path)
        self.map_dict = {}
        # self.build_global_map()
        # self.points_arr =  np.asarray(self.point_cloud.points)
        # Convert to a numpy array and then to a Python list.
    def build_global_map(self):
        offset_arr = np.array([self.global_map_bound[0],self.global_map_bound[2],self.global_map_bound[4]])
        size_x = int((self.global_map_bound[1] - self.global_map_bound[0]) / self.voxel_width)
        size_y = int((self.global_map_bound[3] - self.global_map_bound[2]) / self.voxel_width)
        size_z = int((self.global_map_bound[5] - self.global_map_bound[4]) / self.voxel_width)
        logger.info('voxel size:',[size_x,size_y,size_z])
        voxel_map = VoxelMap(size_x=size_x,size_y=size_y,size_z=size_z,offset=offset_arr,voxel_size=self.voxel_width)
        st_tiem = time.time()
        crop_occupied_coords = np.asarray(self.point_cloud.points)
        logger.info(f'crop_occupied_coords:,{[size_x,size_y,size_z]},{crop_occupied_coords[:3]}')
        voxel_map.set_occupied_array(crop_occupied_coords)
        voxel_map.dilate(radius=self.dilate_radius)
        logger.info(f'set time: {time.time()-st_tiem}')
        self.global_voxelmap = voxel_map
        self.path_searcher = PathSearch(voxel_map)
        
    def build_cur_map(self,points_array,margin=100):
        """Build the current local map."""
        bbox_min, bbox_max = get_crop_box(points_array,self.global_map_bound,margin=margin)
        offset_arr = bbox_min
        cur_bbox = [bbox_min[0],bbox_max[0],bbox_min[1],bbox_max[1],bbox_min[2],bbox_max[2]] 
        cur_bbox = np.array(cur_bbox).astype(np.int32)
        print('cur_bbox:',cur_bbox,)
        temp_cur_name = f"{self.env_name}-m{margin}_v{self.voxel_width}_d{self.dilate_radius}_box{'_'.join(cur_bbox.astype(str).tolist())}.pt"    
        temp_cur_path = os.path.join(self.temp_dir,temp_cur_name)
        
        size_x = ((bbox_max[0] - bbox_min[0]) / self.voxel_width)
        size_y = ((bbox_max[1] - bbox_min[1]) / self.voxel_width)
        size_z = ((bbox_max[2] - bbox_min[2]) / self.voxel_width)
    
        # if os.path.exists(temp_cur_path):
        if temp_cur_path in self.map_dict:
            st_time = time.time()
            # with open(temp_cur_path, 'rb') as f:
                # voxel_map = pickle.load(f)
            voxel_map = self.map_dict[temp_cur_path]
            print(f'loading temp voxmap with key:{temp_cur_path}')
        else:
            voxel_map = VoxelMap(size_x=size_x,size_y=size_y,size_z=size_z,offset=offset_arr,voxel_size=self.voxel_width)
            crop_pcd = crop_pointcloud_with_box(self.point_cloud,bbox_min,bbox_max)
            # print('crop time:',time.time()-st_tiem)
            crop_occupied_coords = np.asarray(crop_pcd.points)
            voxel_map.set_occupied_array(crop_occupied_coords)
            voxel_map.dilate(radius=self.dilate_radius)
            # print('set time:',time.time()-st_tiem)
            # with open(temp_cur_path, 'wb') as f:
                # pickle.dump(voxel_map, f)
            # voxel_map.save_voxels(temp_cur_path)
            self.map_dict[temp_cur_path] = voxel_map
                
            logger.info(f'saving cache voxel_map to: {temp_cur_path}')
            
        path_searcher = PathSearch(voxel_map)
        return voxel_map,path_searcher
    
    def plan_path_direct(self, start_pos: List[float],start_quaternionr: List[float], goal_pos: List[float],) -> Tuple[bool, List[List[float]]]:
        """Run path planning directly."""
        
        point_arr = np.array([start_pos,goal_pos])
        begin_point = np.array(start_pos)
        target_point = np.array(goal_pos)
        pitch, roll, yaw = to_eularian_angles(start_quaternionr[0],start_quaternionr[1],start_quaternionr[2],start_quaternionr[3])
        cur_voxel_map,cur_path_searcher = self.build_cur_map(point_arr,margin=50)
        # if self.path_searcher is None or self.global_voxelmap is None:
        #     self.build_global_map()
        
        path_result = cur_path_searcher.hybrid_a_star(begin_point, target_point,thr=5)     
        # path_result = cur_path_searcher.hybrid_a_star(begin_point, target_point)    
        if len(path_result) > 1:
            record_list_, action_list_ = cur_path_searcher.backtrack_path_with_yaw(path_result,initial_yaw=yaw, with_stop=True)
            return True, (record_list_,action_list_)
        
        logger.warning(f"[{self.task_name}] No path found between start and goal")
        return False,([],[])
    
    def is_initialized(self) -> bool:
        """Check whether initialization has completed."""
        return self.map_initialized and self.pathsearch is not None
    
    def get_current_environment(self) -> str:
        """Get the current environment name."""
        return self.current_map_name
    
    def get_map_bounds(self) -> List[float]:
        """Get map bounds."""
        return self.map_bound.copy()

def load_thread_config(config_file: str) -> List[PlannerConfig]:
    """Load thread configuration."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    configs = []
    if "thread_params" in config:
        for param in config["thread_params"]:
            configs.append(PlannerConfig(param))
    
    return configs

def load_objects_from_json_array(file_path: str) -> List:
    """Load objects from a JSON array file."""
    objects = []
    try:
        with open(file_path, 'r') as file:
            json_array = json.load(file)
        
        # Check whether the top-level value is an array.
        if not isinstance(json_array, list):
            print("JSON file is not an array")
            return objects
        
        # Iterate over each object in the array.
        for json_object in json_array:
            objects.append(json_object)
        
    except Exception as e:
        print(f"JSON parsing error: {e}")
    
    return objects

def split_pos_list_by_rank(pos_list: List, total: int, rank: int) -> List:
    """Split a position list by rank."""
    # Validate arguments.
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
    
    # Return the subset.
    return pos_list[start_idx:end_idx]

def run_path_planner_thread(plan_objects: List,task_name:str,env_name:str,output_dir:str=""):
    """Run a path-planner thread."""
    
    # try:
    path_planner = PathPlannerNode(task_name, env_name)
    print(f"\033[32mStarting path planning for thread: {task_name}\033[0m")
    print(f"\033[36mProcessing {len(plan_objects)} planning objects\033[0m")
    # Process each planning object.
    success_count = 0
    failed_count = 0
    env_output_dir = os.path.join(output_dir,env_name)
    os.makedirs(env_output_dir,exist_ok=True)
    result_list = []
    
    plan_obj_list : List[ONPlanObject] = []
    for plan_obj in plan_objects:
        for j, pose in enumerate(plan_obj['pose']):
            plan_obj_list.append(
                ONPlanObject(
                    episode_id=plan_obj['episode_id'],
                    pose_idx = j,
                    start_position=plan_obj["start_pose"]['start_position'],
                    start_quaternion=plan_obj["start_pose"]['start_quaternionr'],
                    target_position=pose
                )
            )
    plan_obj_list = sorted(plan_obj_list,key=lambda row: (tuple(row.start_position),tuple(row.target_position)))
    print(f'examples:',[(row.episode_id,row.pose_idx)for row in plan_obj_list[:5]])
    for obj in tqdm(plan_obj_list,desc=f'[{task_name}-{env_name}]'):
        # try:
            if len(obj.start_position) < 3:
                print(f"[{task_name}] Invalid start position for eps {obj.episode_id}")
                failed_count += 1
                continue
            start_pos = obj.start_position  # First 3 coordinates.
            start_quaternion = obj.start_quaternion  # First 3 coordinates.
            
            # for j, pose in enumerate(obj.poses):
            if len(obj.target_position) < 3:
                print(f"[{task_name}] Invalid pose {obj.pose_idx}: { obj.target_position} for object {obj.episode_id}")
                continue
            
            goal_pos =  obj.target_position[:3]  # First 3 coordinates.
            
            logger.info(f'Eps id:{obj.episode_id} Planning for: {start_pos} - > {goal_pos}')
            
            planning_result, (record_list,action_list) = path_planner.plan_path_direct(
                start_pos,start_quaternion, goal_pos)
            
            logger.info(f'Eps id:{obj.episode_id} Planning Status: {planning_result}, Waypoint: {len(record_list)}')
            
            if planning_result:
                success_count += 1
                if env_output_dir !="":
                    filepath = os.path.join(env_output_dir,f"{obj.episode_id}", f"{obj.pose_idx}.json")
                    save_path_result(filepath,record_list,action_list,start_pos, goal_pos)
            else:
                failed_count += 1
                
            result_list.append({
                'episode_id':obj.episode_id,
                'pose_idx': obj.pose_idx,
                'success': planning_result,
                'filepath': filepath if planning_result else ""
            })
        # except Exception as e:
        #     failed_count += 1
        #     print(f"[{config.name}] Error processing object {i + 1}: {e}")
    with open(os.path.join(output_dir,f'{env_name}_result.json'),'w') as f:
        json.dump({
            'total': len(plan_objects),
            'success': success_count,
            'failed': failed_count,
            'record': result_list
        },f,indent=2)
    print(f"\033[36m {task_name} completed! Results: {success_count} successful, {failed_count} failed")

def save_path_result(output_path: str, record_list: List,action_list: List,start_pos, goal_pos):
    """Save path results."""
    
    act_type_list  = [action[0][0] for action in action_list]
    act_value_list  = [action[0][1] for action in action_list]
    pos_list = [action[1].tolist() for action in action_list]
    yaws = [record[3] for record in record_list]
    
    record_list_save = [row.tolist() for row in record_list]
    action_list_save = [(row[0],row[1].tolist()) for row in action_list]
    
    result_data = {
        "image_path": "",
        "gpt_instruction": "",
        "record_list": record_list_save,
        "action_list": action_list_save,
        "index_list": [],
        "action_type": act_type_list,
        "action": act_value_list,
        "pos": pos_list,
        "yaw": yaws,
        "start_pos":start_pos, 
        "goal_pos":goal_pos
    }

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)

def get_project_directory() -> str:
    """Get the project directory."""
    current_path = Path(__file__).resolve()
    # Adjust according to the actual project structure.
    project_dir = current_path.parent.parent
    return str(project_dir)

def main():
    """Main entry point."""
    # Get the project directory.
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_airsim_16', help="input env name")
    args = parser.parse_args()
    
    project_dir = get_project_directory()
    print(f"Project directory: {project_dir}")
    # Get the environment name.
    env_name = args.env
    if "ENV" in os.environ:
        print(f"\033[32mYour ENV is: {env_name}\033[0m")
    else:
        print(f"\033[33mUse default environment: {env_name}\033[0m")
    
    # Load configuration.
    config_path = os.path.join(project_dir, "configs", f"{env_name}.yaml")
    configs = load_thread_config(config_path)
    print(f"Config name: {configs[0].name}, aim port: {configs[0].aim_port}")
    
    # Load data.
    data_path = os.path.join(project_dir, "data", "dataset", f"{env_name}_train.json")
    objects = load_objects_from_json_array(data_path)
    
    output_save_path = "output-v2/"
    env_base_json_folder = os.path.join(output_save_path,env_name)
    cur_objects = []
    if os.path.exists(env_base_json_folder):
        eps_folders = os.listdir(env_base_json_folder)
        for obj in objects:
            if f"{obj['episode_id']}" in eps_folders:
                continue
            cur_objects.append(obj)
    else:
        cur_objects = objects
    print(f"Loading json data: {data_path}")
    print(f"Loaded {len(cur_objects)}/{len(objects)} objects")
    
    if len(cur_objects) > 0:
        # Create threads.
        threads = []
        for i, config in enumerate(configs):
            # Split data.
            cur_list = split_pos_list_by_rank(cur_objects, len(configs), i)  # Hard-coded here; adjust if needed.
            # cur_list = objects
            print(f"cur list: {len(cur_list)} total: {len(objects)}")
            output_dir=os.path.join(get_project_directory(),output_save_path)
            # Start the thread.
            thread = threading.Thread(
                target=run_path_planner_thread,
                args=(cur_list, f"Thread {i}", env_name, output_dir)
            )
            threads.append(thread)
            thread.start()
            print(f"\033[33mStarted thread: {config.name}\033[0m")
        # Wait for all threads to complete.
        for thread in threads:
            thread.join()
        print("All threads completed!")

if __name__ == "__main__":
    main()
