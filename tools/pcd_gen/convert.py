import open3d as o3d
import argparse
import os
import numpy as np

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
def main():
    parser = argparse.ArgumentParser(description="将PCD文件转换为PLY格式")
    parser.add_argument('--input', type=str, help="输入的PCD文件路径")
    parser.add_argument('--output', type=str, help="输出的PLY文件路径")
    
    args = parser.parse_args()
    
    # Check whether the input file exists.
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        return
    
    # Check the input file extension.
    if not args.input.lower().endswith('.pcd'):
        print(f"警告: 输入文件可能不是PCD格式 - {args.input}")
    
    # Ensure the output file has the correct extension.
    if not args.output.lower().endswith('.ply'):
        args.output += '.ply'
        print(f"自动添加.ply扩展名: {args.output}")
    
    # Create the output directory if it does not exist.
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    # Run the conversion.
    print(f"转换: {args.input} -> {args.output}")
    pcd_to_ply(args.input, args.output)

if __name__ == "__main__":
    main()
