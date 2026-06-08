import cosysairsim as airsim
import json
import numpy as np
import cv2
import os
import time
import math
from datetime import datetime
from typing import List, Dict


class AirSim360DataCollector:
    def __init__(self, positions_file="saved_positions.json"):
        self.client = None
        self.positions_file = positions_file
        self.positions = []
        self.connected = False
        self.IMAGE_WIDTH = 1280
        self.IMAGE_HEIGHT = 720
        self.img_cnt = 0

    def connect_to_airsim(self):
        """Connect to AirSim."""
        try:
            self.client = airsim.ComputerVisionClient()
            self.client.confirmConnection()
            print("成功连接到AirSim")
            self.connected = True
            return True
        except Exception as e:
            print(f"连接AirSim失败: {e}")
            self.connected = False
            return False

    def load_positions(self):
        """Load position information from a JSON file."""
        try:
            with open(self.positions_file, "r", encoding="utf-8") as f:
                self.positions = json.load(f)
            print(f"成功加载 {len(self.positions)} 个位置")
            return True
        except FileNotFoundError:
            print(f"位置文件 {self.positions_file} 不存在")
            return False
        except Exception as e:
            print(f"加载位置文件失败: {e}")
            return False

    def set_cam_position(self, position_data):
        """Move to the specified position."""
        try:
            position = position_data["position"]

            # Create the target pose.
            target_pose = airsim.Pose()
            target_pose.position.x_val = position[0]
            target_pose.position.y_val = position[1]
            target_pose.position.z_val = position[2]

            # Set the default orientation, facing forward.
            target_pose.orientation.w_val = 1.0
            target_pose.orientation.x_val = 0.0
            target_pose.orientation.y_val = 0.0
            target_pose.orientation.z_val = 0.0

            # Set the camera position.
            self.client.simSetCameraPose("front_center", target_pose)

            # Wait for motion to finish.
            time.sleep(0.5)

            print(
                f"已移动到位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
            return True

        except Exception as e:
            print(f"移动到位置失败: {e}")
            return False

    def set_camera_orientation(self, yaw_deg, pitch_deg, position):
        """Set camera orientation."""
        try:
            # Convert degrees to radians.
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            roll_rad = 0  # Keep level.

            # Create the new camera pose.
            new_pose = airsim.Pose()
            new_pose.position.x_val = position[0]
            new_pose.position.y_val = position[1]
            new_pose.position.z_val = position[2]

            # Compute the quaternion from ZYX Euler angles.
            cy = math.cos(yaw_rad * 0.5)
            sy = math.sin(yaw_rad * 0.5)
            cp = math.cos(pitch_rad * 0.5)
            sp = math.sin(pitch_rad * 0.5)
            cr = math.cos(roll_rad * 0.5)
            sr = math.sin(roll_rad * 0.5)

            new_pose.orientation.w_val = cy * cp * cr + sy * sp * sr
            new_pose.orientation.x_val = cy * cp * sr - sy * sp * cr
            new_pose.orientation.y_val = sy * cp * sr + cy * sp * cr
            new_pose.orientation.z_val = sy * cp * cr - cy * sp * sr

            # Set camera pose.
            self.client.simSetCameraPose("front_center", new_pose)

            return True

        except Exception as e:
            print(f"设置相机朝向失败: {e}")
            return False

    def get_camera_info(self):
        """Get camera information."""
        try:
            camera_info = self.client.simGetCameraInfo("front_center")
            return camera_info
        except Exception as e:
            print(f"获取相机信息失败: {e}")
            return None

    def save_camera_state(self, filename, camera_dir):
        """Save camera state information to a JSON file."""
        try:
            # Get camera information.
            camera_info = self.get_camera_info()
            if camera_info is None:
                print(f"无法获取相机信息，跳过状态保存: {filename}")
                return False

            # Extract position information.
            position = camera_info.pose.position
            orientation = camera_info.pose.orientation

            # Extract the projection matrix.
            proj_mat = camera_info.proj_mat.matrix

            # Build state data.
            camera_data = {
                "position": [
                    float(position.x_val),
                    float(position.y_val),
                    float(position.z_val),
                ],
                "orientation": [
                    float(orientation.w_val),
                    float(orientation.x_val),
                    float(orientation.y_val),
                    float(orientation.z_val),
                ],
                "projection_matrix": [
                    [
                        float(proj_mat[0][0]),
                        float(proj_mat[0][1]),
                        float(proj_mat[0][2]),
                        float(proj_mat[0][3]),
                    ],
                    [
                        float(proj_mat[1][0]),
                        float(proj_mat[1][1]),
                        float(proj_mat[1][2]),
                        float(proj_mat[1][3]),
                    ],
                    [
                        float(proj_mat[2][0]),
                        float(proj_mat[2][1]),
                        float(proj_mat[2][2]),
                        float(proj_mat[2][3]),
                    ],
                    [
                        float(proj_mat[3][0]),
                        float(proj_mat[3][1]),
                        float(proj_mat[3][2]),
                        float(proj_mat[3][3]),
                    ],
                ],
                "fov": float(camera_info.fov),
                "timestamp": datetime.now().isoformat(),
            }

            # Ensure the directory exists.
            os.makedirs(camera_dir, exist_ok=True)

            # Save to file.
            camera_file_path = os.path.join(camera_dir, filename)
            with open(camera_file_path, "w", encoding="utf-8") as f:
                json.dump(camera_data, f, indent=4, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"保存相机状态失败: {e}")
            return False

    def capture_images_at_orientation(self, yaw, pitch):
        """Capture images at the specified orientation."""
        try:
            # Build image requests.
            image_requests = [
                airsim.ImageRequest(
                    "front_center", airsim.ImageType.Scene, False, False
                ),
                airsim.ImageRequest(
                    "front_center", airsim.ImageType.DepthPlanar, True, False
                ),
            ]

            # Get image data.
            responses = self.client.simGetImages(image_requests)

            if len(responses) < 2:
                print("获取图像响应不完整")
                return None, None

            # Process the color image.
            img_data = None
            if len(responses[0].image_data_uint8) > 0:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                if len(img_1d) >= self.IMAGE_HEIGHT * self.IMAGE_WIDTH * 3:
                    img_data = img_1d.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
                    img_data = cv2.resize(
                        img_data, (640, 360), interpolation=cv2.INTER_LINEAR
                    )

            # Process the depth image.
            depth_data = None
            if len(responses[1].image_data_float) > 0:
                depth_1d = np.array(responses[1].image_data_float, dtype=np.float32)
                if len(depth_1d) >= self.IMAGE_HEIGHT * self.IMAGE_WIDTH:
                    depth_data = depth_1d.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
                    depth_data = cv2.resize(
                        depth_data, (640, 360), interpolation=cv2.INTER_NEAREST
                    )

            return img_data, depth_data

        except Exception as e:
            print(f"捕获图像失败 (yaw={yaw}, pitch={pitch}): {e}")
            return None, None

    def collect_360_data_at_position(
        self, position_index, position_data, save_dir="360_data"
    ):
        """Collect 360-degree data at the specified position."""
        if not self.connected:
            print("未连接到AirSim")
            return False

        print(f"\n开始处理位置 {position_index + 1}/{len(self.positions)}")
        print(f"位置坐标: {position_data['position']}")
        print(f"时间戳: {position_data['timestamp']}")

        # Move to the target position.
        if not self.set_cam_position(position_data):
            print(f"无法移动到位置 {position_index}")
            return False

        # Create save directories.
        image_dir = os.path.join(save_dir, "images")
        depth_dir = os.path.join(save_dir, "depths")
        camera_dir = os.path.join(save_dir, "cameras")  # Additional state directory.

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)

        # Define rotation parameters.
        yaw_step = 30  # Horizontal step: every 30 degrees.
        pitch_step = 20  # Pitch step: every 30 degrees.

        yaw_angles = list(range(0, 360, yaw_step))
        pitch_angles = list(range(-45, 46, pitch_step))

        total_captures = len(yaw_angles) * len(pitch_angles)
        capture_count = 0
        successful_captures = 0

        print(f"开始采集 {total_captures} 张图像...")

        for yaw in yaw_angles:
            for pitch in pitch_angles:
                # Set camera orientation.
                if self.set_camera_orientation(yaw, pitch, position_data["position"]):
                    # Wait for the camera to stabilize.
                    time.sleep(1)

                    # Capture images.
                    img_data, depth_data = self.capture_images_at_orientation(
                        yaw, pitch
                    )

                    if img_data is not None and depth_data is not None:
                        # Build file name.
                        filename_base = f"{self.img_cnt:06d}"

                        # Save the color image.
                        img_filename = os.path.join(image_dir, f"{filename_base}.png")
                        cv2.imwrite(
                            img_filename, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        )

                        # Save the depth image.
                        depth_filename = os.path.join(depth_dir, f"{filename_base}.png")
                        depth_img_uint16 = (depth_data * 100).astype(np.uint16)
                        cv2.imwrite(depth_filename, depth_img_uint16)

                        # Save camera state.
                        camera_filename = f"{filename_base}.json"
                        if self.save_camera_state(camera_filename, camera_dir):
                            successful_captures += 1
                        else:
                            print(
                                f"警告: 图像 {filename_base} 保存成功，但状态信息保存失败"
                            )
                            successful_captures += 1  # Still count as success because images were saved.

                        self.img_cnt += 1
                    else:
                        print(f"图像捕获失败 (yaw={yaw}, pitch={pitch})")
                else:
                    print(f"设置相机朝向失败 (yaw={yaw}, pitch={pitch})")

                capture_count += 1

                # Progress display.
                if capture_count % 100 == 0:
                    progress = (capture_count / total_captures) * 100
                    print(f"  进度: {progress:.1f}% ({capture_count}/{total_captures})")

        print(
            f"  位置 {position_index} 完成! 成功采集 {successful_captures}/{total_captures} 张图像"
        )
        return True

    def collect_all_positions(self, save_dir="360_data", start_from=0, end_at=None):
        """Collect 360-degree data for all positions."""
        if not self.positions:
            print("没有加载位置数据")
            return False

        end_index = end_at if end_at is not None else len(self.positions)
        end_index = min(end_index, len(self.positions))

        print(f"开始采集位置 {start_from} 到 {end_index-1} 的360度数据")
        print(f"总共 {end_index - start_from} 个位置")

        success_count = 0

        for i in range(start_from, end_index):
            try:
                if self.collect_360_data_at_position(i, self.positions[i], save_dir):
                    success_count += 1
                else:
                    print(f"位置 {i} 采集失败")

                # Rest briefly between positions.
                time.sleep(2)

            except KeyboardInterrupt:
                print(f"\n用户中断，已完成 {success_count} 个位置的采集")
                break
            except Exception as e:
                print(f"位置 {i} 发生错误: {e}")
                continue

        print(f"\n采集完成! 成功处理 {success_count}/{end_index - start_from} 个位置")
        return True

    def run(self, save_dir="360_data", start_from=0, end_at=None):
        """Main run function."""
        print("=== AirSim 360度数据采集器 ===")

        # Load position data.
        if not self.load_positions():
            return False

        # Connect to AirSim.
        if not self.connect_to_airsim():
            print("无法连接到AirSim，请确保AirSim正在运行")
            return False

        # Start collection.
        try:
            self.collect_all_positions(save_dir, start_from, end_at)
        except Exception as e:
            print(f"采集过程中发生错误: {e}")
        finally:
            print("程序结束")


def main():
    # Configuration parameters.
    positions_file = "saved_positions.json"  # Position file path.
    save_directory = "mapping_data"  # Save directory.
    start_position = 0  # Start position index.
    end_position = None  # End position index; None means all positions.

    # Create the collector and run it.
    collector = AirSim360DataCollector(positions_file)
    collector.run(save_directory, start_position, end_position)


if __name__ == "__main__":
    main()
