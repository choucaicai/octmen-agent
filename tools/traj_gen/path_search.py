import numpy as np
import heapq
import math
import torch
import time
from typing import List, Tuple, Optional, Dict, Any
import os
from dataclasses import dataclass, field
import copy
# Simple voxel map implementation.
class VoxelMap:
    def __init__(self, 
                 size_x: int, size_y: int, size_z: int,
                 offset: np.ndarray, 
                 voxel_size: float,
                 occupied_voxels=None,
                 ):
        """
        Initialize the voxel map.
        Args:
            size_x, size_y, size_z: voxel map size along each axis
            offset: offset array with shape (3,), the map origin in world coordinates
            voxel_size: voxel size, assuming the same size for x/y/z
        """
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.offset_arr = np.array(offset, dtype=np.float64)
        self.voxel_width = voxel_size
        
        # Store occupied voxel coordinates in a set.
        if occupied_voxels is None:
            self.occupied_voxels = set()
        else:
            self.occupied_voxels = copy.deepcopy(occupied_voxels)
            
    def world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert world coordinates to voxel coordinates.
        """
        voxel_coords = ((point - self.offset_arr) / self.voxel_width).astype(np.int32)
        return tuple(voxel_coords)
    
    def voxel_to_world(self, voxel_coords: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel coordinates to world coordinates at the voxel center.
        """
        voxel_array = np.array(voxel_coords, dtype=np.float64)
        world_point = voxel_array * self.voxel_width + self.offset_arr + self.voxel_width / 2
        return world_point
    
    def is_valid_voxel(self, voxel_coords: Tuple[int, int, int]) -> bool:
        """
        Check whether voxel coordinates are within the valid range.
        """
        x, y, z = voxel_coords
        return (0 <= x < self.size_x and 
                0 <= y < self.size_y and 
                0 <= z < self.size_z)
    
    def query(self, pos: np.ndarray) -> bool:
        """
        Query whether a position is occupied.
        """
        voxel_coords = self.world_to_voxel(pos)
        # Check whether coordinates are within the valid range.
        if not self.is_valid_voxel(voxel_coords):
            return True  # Treat out-of-bounds positions as occupied.
            
        return voxel_coords in self.occupied_voxels
    
    def set_occupied(self, pos: np.ndarray, occupied: bool = True):
        """
        Set the occupancy state for a position.
        """
        try:
            voxel_coords = self.world_to_voxel(pos)
            if not self.is_valid_voxel(voxel_coords):
                print(f"Warning: Trying to set voxel outside valid range: {voxel_coords}")
                return
            if occupied:
                self.occupied_voxels.add(voxel_coords)
            else:
                self.occupied_voxels.discard(voxel_coords)
                
        except (ValueError, OverflowError):
            print(f"Warning: Invalid position for voxel conversion: {pos}")
    
    def set_occupied_voxel(self, voxel_coords: Tuple[int, int, int], occupied: bool = True):
        """
        Set occupancy directly by voxel coordinates.
        """
        if not self.is_valid_voxel(voxel_coords):
            print(f"Warning: Trying to set voxel outside valid range: {voxel_coords}")
            return
        if occupied:
            self.occupied_voxels.add(voxel_coords)
        else:
            self.occupied_voxels.discard(voxel_coords)
            
    def save_voxels(self,save_path):
        # coords_array = np.array(list(self.occupied_voxels), dtype=np.int32)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        torch.save(
            dict(
                occupied_coords=self.occupied_voxels,
                offset_arr=self.offset_arr,
                size_x = self.size_x,
                size_y = self.size_y,
                size_z = self.size_z,
                voxel_width = self.voxel_width
            ),save_path)
        # data = np.load('voxel_data.npz')
        # coords_array = data['occupied_coords']

    def set_occupied_array(self, coord_array:np.array ):
        voxel_coords_arr = ((coord_array - self.offset_arr) / self.voxel_width).astype(np.int32)
        valid_mask = (
            (voxel_coords_arr[:, 0] >= 0) & (voxel_coords_arr[:, 0] < self.size_x) &
            (voxel_coords_arr[:, 1] >= 0) & (voxel_coords_arr[:, 1] < self.size_y) &
            (voxel_coords_arr[:, 2] >= 0) & (voxel_coords_arr[:, 2] < self.size_z)
        )
        print(voxel_coords_arr[:3],self.offset_arr)
        valid_voxel_coords = voxel_coords_arr[valid_mask]
        # print('map tuple:',list(map(tuple, valid_voxel_coords))[:3])
        self.occupied_voxels = set(map(tuple, valid_voxel_coords))
        print(f'Set VoxelMap successfully, Size: [{self.size_x},{self.size_y},{self.size_z}],Offset:{self.offset_arr.tolist()},Voxels: {len(self.occupied_voxels)}')
        
    def get_occupied_count(self) -> int:
        """Get the number of occupied voxels."""
        return len(self.occupied_voxels)
    
    def clear(self):
        """Clear all occupied voxels."""
        self.occupied_voxels.clear()
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get map bounds in world coordinates.
        """
        min_bound = self.offset_arr
        max_bound = self.offset_arr + np.array([self.size_x, self.size_y, self.size_z]) * self.voxel_width
        return min_bound, max_bound
    
    def is_point_in_bounds(self, point: np.ndarray) -> bool:
        """
        Check whether a point is within map bounds.
        """
        min_bound, max_bound = self.get_bounds()
        return np.all(point >= min_bound) and np.all(point < max_bound)
    
    def dilate(self, radius: int = 1):
        """
        Dilate occupied voxels using a vectorized implementation.
        Args:
            radius: dilation radius in voxels
        """
        if radius <= 0:
            return
        if not self.occupied_voxels:
            return
        # Convert to a numpy array.
        occupied_arr = np.array(list(self.occupied_voxels))
        
        # Generate the offset grid.
        offsets = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1].reshape(3, -1).T
        
        # Compute all neighbor coordinates with broadcasting.
        neighbors = occupied_arr[:, None, :] + offsets[None, :, :]
        neighbors = neighbors.reshape(-1, 3)
        
        # Vectorized bounds check.
        valid_mask = (
            (neighbors[:, 0] >= 0) & (neighbors[:, 0] < self.size_x) &
            (neighbors[:, 1] >= 0) & (neighbors[:, 1] < self.size_y) &
            (neighbors[:, 2] >= 0) & (neighbors[:, 2] < self.size_z)
        )
        valid_neighbors = neighbors[valid_mask]
        self.occupied_voxels = set(map(tuple, valid_neighbors))
    
@dataclass
class Node:
    pos: np.ndarray
    yaw: float
    g_cost: float
    h_cost: float
    parent: Optional['Node'] = None
    
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

class PathSearch:
    def __init__(self, global_map: VoxelMap):
        self.global_map = global_map
        self.end_ = None
        self.record_list_ = []
        self.action_list_ = []
    
    def calculate_yaw(self, direction: np.ndarray) -> float:
        """Calculate the yaw angle for a direction vector."""
        return math.atan2(direction[1], direction[0])
    
    def manhattan(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Manhattan distance."""
        return np.sum(np.abs(v1 - v2))
    
    def heuristic(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Euclidean-distance heuristic."""
        return np.linalg.norm(v1 - v2)
    
    def cal_dis(self, start, end) -> float:
        """Calculate distance."""
        if isinstance(start, Node):
            start_pos = start.pos
        else:
            start_pos = start
            
        if isinstance(end, Node):
            end_pos = end.pos
        else:
            end_pos = end
            
        return np.linalg.norm(start_pos - end_pos)
    
    def is_occupied(self, pos: np.ndarray) -> bool:
        """Check whether a position is occupied."""
        return self.global_map.query(pos)
    
    def get_neighbors(self, current: Node) -> List[Node]:
        """Get neighbor nodes for the current node."""
        neighbors = []
        cur_node_pos = current.pos
        cur_yaw = current.yaw
        
        # Define motion directions.
        angles = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]
        step_size = 3
        
        motions = []
        for angle in angles:
            rad = math.radians(angle)
            motions.append(np.array([step_size * math.cos(rad), 
                                   step_size * math.sin(rad), 0]))
        
        # Add vertical motion.
        motions.append(np.array([0, 0, 3]))   # Ascend.
        motions.append(np.array([0, 0, -3]))  # Descend.
        
        for i, motion in enumerate(motions):
            tmp_pos = cur_node_pos + motion
            
            # Compute the angle cost.
            if i < len(angles) and angles[i] != cur_yaw:
                angle_cost = 2.0
            else:
                angle_cost = 0
            
            if not self.is_occupied(tmp_pos):
                if i < len(angles):
                    new_yaw = math.radians(angles[i])
                else:
                    new_yaw = cur_yaw
                
                tmp_gcost = self.heuristic(cur_node_pos, tmp_pos)
                h_cost = self.heuristic(tmp_pos, self.end_)
                
                neighbor = Node(
                    pos=tmp_pos,
                    yaw=new_yaw,
                    g_cost=current.g_cost + tmp_gcost + angle_cost,
                    h_cost=h_cost,
                    parent=current
                )
                neighbors.append(neighbor)
        
        return neighbors
    
    def hybrid_a_star(self, start: np.ndarray, end: np.ndarray,thr = 3.1) -> List[np.ndarray]:
        """Hybrid A* path search."""
        self.end_ = end
        open_list = []
        closed_list = {}
        # thr = 3.1
        
        start_node = Node(
            pos=start,
            yaw=0,
            g_cost=0,
            h_cost=self.heuristic(start, end)
        )
        
        heapq.heappush(open_list, start_node)
        
        if self.is_occupied(start_node.pos):
            print("Start point is occupied!")
            return []
        
        start_time = time.time()
        timeout_duration = 600  # 2-minute timeout.
        
        while open_list:
            if time.time() - start_time > timeout_duration:
                print("\033[33mSearch timeout exceeded 2 minutes.\033[0m")
                return []
            
            current = heapq.heappop(open_list)
            
            # Check whether the goal is reached.
            if self.heuristic(current.pos, end) < thr:
                path = []
                while current is not None:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                return path
            
            # Get neighbor nodes.
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                # Create a hash key.
                neighbor_hash = (int(neighbor.pos[0]) * 1000000 + 
                               int(neighbor.pos[1]) * 1000 + 
                               int(neighbor.pos[2]))
                
                if neighbor_hash in closed_list:
                    continue
                
                neighbor.h_cost = self.heuristic(neighbor.pos, end)
                heapq.heappush(open_list, neighbor)
                closed_list[neighbor_hash] = neighbor
        
        return []
    
    def calculate_record(self, cur_p: np.ndarray, next_p: np.ndarray, 
                        in_yaw: float, out_yaw: float) -> List[np.ndarray]:
        """Calculate the record list."""
        record_list = []
        yaw_error = round((out_yaw - in_yaw) * 180 / math.pi)
        
        if yaw_error < -180:
            yaw_error += 360
        if yaw_error > 180:
            yaw_error -= 360
        
        turn_nums = abs(round(yaw_error / 30.0))
        step_yaw = math.pi / 180 * 30
        
        if yaw_error != 0:
            if yaw_error > 0:
                for i in range(turn_nums + 1):
                    record_list.append(np.array([cur_p[0], cur_p[1], cur_p[2], 
                                               in_yaw + i * step_yaw]))
            else:
                for i in range(turn_nums + 1):
                    record_list.append(np.array([cur_p[0], cur_p[1], cur_p[2], 
                                               in_yaw - i * step_yaw]))
        else:
            record_list.append(np.array([cur_p[0], cur_p[1], cur_p[2], in_yaw]))
        
        return record_list
    
    def calculate_action(self, start: np.ndarray, end: np.ndarray, 
                        in_yaw: float, out_yaw: float) -> List[Tuple[Tuple[str, int], np.ndarray]]:
        """Calculate the action list."""
        action_list = []
        yaw_error = round((out_yaw - in_yaw) * 180 / math.pi)
        
        if yaw_error < -180:
            yaw_error += 360
        if yaw_error > 180:
            yaw_error -= 360
        
        turn_nums = abs(round(yaw_error / 30.0))
        
        # Handle vertical motion.
        if end[2] - start[2] > 1:
            action_list.append((("go up", round(self.cal_dis(start, end))), start))
            return action_list
        
        if end[2] - start[2] < -1:
            action_list.append((("go down", round(self.cal_dis(start, end))), start))
            return action_list
        # Handle turning.
        if yaw_error != 0:
            if yaw_error > 0:
                for i in range(turn_nums):
                    action_list.append((("turn left", 30), start))
            else:
                for i in range(turn_nums):
                    action_list.append((("turn right", 30), start))
            
            # Go straight after turning.
            action_list.append((("go straight", round(self.cal_dis(start, end))), start))
        else:
            # No turn is needed; go straight directly.
            action_list.append((("go straight", round(self.cal_dis(start, end))), start))
        
        return action_list
    
    def backtrack_path_inplace(self, path: List[np.ndarray], with_stop: bool = False):
        """Backtrack the path in place."""
        self.record_list_.clear()
        self.action_list_.clear()
        
        if len(path) <= 2:
            print("Path error: Path size is too small.")
            return
        
        start_yaw = self.calculate_yaw(path[1] - path[0])
        for i in range(len(path) - 1):
            out_yaw = self.calculate_yaw(path[i + 1] - path[i])
            
            if i == 0:
                in_yaw = start_yaw
            else:
                in_yaw = self.calculate_yaw(path[i] - path[i - 1])
                if abs(path[i + 1][2] - path[i][2]) > 1:
                    in_yaw = self.record_list_[-1][3]
                    out_yaw = in_yaw
                elif abs(path[i][2] - path[i - 1][2]) > 1:
                    in_yaw = self.record_list_[-1][3]
            
            tmp_act_list = self.calculate_action(path[i], path[i + 1], in_yaw, out_yaw)
            tmp_rec_list = self.calculate_record(path[i], path[i + 1], in_yaw, out_yaw)
            
            self.action_list_.extend(tmp_act_list)
            self.record_list_.extend(tmp_rec_list)
        
        if with_stop and self.record_list_:
            end_yaw = self.record_list_[-1][3]
            self.record_list_.append(np.array([path[-1][0], path[-1][1], path[-1][2], end_yaw]))
            self.action_list_.append((("stop", 0), path[-1]))
            
    def backtrack_path_inplace_with_yaw(self, path: List[np.ndarray], initial_yaw: float, with_stop: bool = False):
        """Backtrack the path in place using the initial yaw."""
        self.record_list_.clear()
        self.action_list_.clear()
        
        if len(path) <= 2:
            print("Path error: Path size is too small.")
            return
        for i in range(len(path) - 1):
            out_yaw = self.calculate_yaw(path[i + 1] - path[i])
            if i == 0:
                in_yaw = initial_yaw
            else:
                in_yaw = self.calculate_yaw(path[i] - path[i - 1])
                if abs(path[i + 1][2] - path[i][2]) > 1:
                    in_yaw = self.record_list_[-1][3]
                    out_yaw = in_yaw
                elif abs(path[i][2] - path[i - 1][2]) > 1:
                    in_yaw = self.record_list_[-1][3]
            
            tmp_act_list = self.calculate_action(path[i], path[i + 1], in_yaw, out_yaw)
            tmp_rec_list = self.calculate_record(path[i], path[i + 1], in_yaw, out_yaw)
            
            self.action_list_.extend(tmp_act_list)
            self.record_list_.extend(tmp_rec_list)
        
        if with_stop and self.record_list_:
            end_yaw = self.record_list_[-1][3]
            self.record_list_.append(np.array([path[-1][0], path[-1][1], path[-1][2], end_yaw]))
            self.action_list_.append((("stop", 0), path[-1]))
            
    def backtrack_path(self, path: List[np.ndarray], with_stop: bool = False) -> Tuple[List[np.ndarray], List[Tuple[Tuple[str, int], np.ndarray]]]:
        """Backtrack the path and return the result."""
        self.backtrack_path_inplace(path, with_stop)
        return self.record_list_.copy(), self.action_list_.copy()
    
    def backtrack_path_with_yaw(self, path: List[np.ndarray], initial_yaw: float, with_stop: bool = False,) -> Tuple[List[np.ndarray], List[Tuple[Tuple[str, int], np.ndarray]]]:
        """Backtrack the path and return the result."""
        self.backtrack_path_inplace_with_yaw(path,initial_yaw=initial_yaw, with_stop=with_stop)
        return self.record_list_.copy(), self.action_list_.copy()
    
    def get_record_list(self) -> List[np.ndarray]:
        """Get the record list."""
        return self.record_list_.copy()
    
    def get_action_list(self) -> List[Tuple[Tuple[str, int], np.ndarray]]:
        """Get the action list."""
        return self.action_list_.copy()

# Usage example.
if __name__ == "__main__":
    # Create a voxel map.
    voxel_map = VoxelMap()
    
    # Set a few obstacles.
    voxel_map.set_occupied(np.array([5, 5, 0]))
    voxel_map.set_occupied(np.array([6, 5, 0]))
    voxel_map.set_occupied(np.array([7, 5, 0]))
    
    # Create the path searcher.
    path_searcher = PathSearch(voxel_map)
    
    # Define start and end points.
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([10.0, 10.0, 0.0])
    
    # Run path search.
    print("开始路径搜索...")
    path = path_searcher.hybrid_a_star(start, end)
    
    if path:
        print(f"找到路径，包含 {len(path)} 个点")
        print("路径点:")
        for i, point in enumerate(path):
            print(f"  {i}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
        
        # Generate action and record lists.
        record_list, action_list = path_searcher.backtrack_path(path, with_stop=True)
        
        print(f"\n动作序列 ({len(action_list)} 个动作):")
        for i, ((action, value), pos) in enumerate(action_list):
            print(f"  {i}: {action} {value} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            
    else:
        print("未找到路径")
