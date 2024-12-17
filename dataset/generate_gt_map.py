import numpy as np
import open3d as o3d
import os
import argparse

from tqdm import tqdm


def load_calibration(calib_file):
    """Load the camera-to-LiDAR calibration matrix."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith("Tr:"):  # Camera-to-LiDAR matrix
                values = list(map(float, line.strip().split()[1:]))
                T_cam2velo = np.eye(4)
                T_cam2velo[:3, :4] = np.array(values).reshape(3, 4)
                return T_cam2velo
    raise RuntimeError("Calibration matrix not found in the file!")


def load_poses(file_path):
    """Load KITTI ground truth poses."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pose file not found: {file_path}")
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.eye(4)  # Initialize a 4x4 identity matrix
            values = list(map(float, line.strip().split()))
            T[:3, :4] = np.array(values).reshape(3, 4)  # Fill rotation + translation
            poses.append(T)
    return poses


def load_lidar_scan(scan_path):
    """Load a single KITTI LiDAR scan."""
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"LiDAR scan file not found: {scan_path}")
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]  # Use only x, y, z (ignore intensity)
    return points


def visualize_point_cloud(points):
    """Visualize a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def arg_parse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a ground truth LiDAR map from KITTI dataset.")
    parser.add_argument('--pose', type=str, default="/home/junseo/datasets/kitti_odometry/gt_pose/dataset/poses/00.txt", help="Path to KITTI pose file.")
    parser.add_argument('--velodyne', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_velodyne/dataset/sequences/00/velodyne", help="Path to KITTI LiDAR scans folder.")
    parser.add_argument('--calib', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_calib/dataset/sequences/00/calib.txt", help="Path to calibration file.")
    parser.add_argument('--output', type=str, default="/home/junseo/MPIL/implementations/lidar_sr/map/gt_map.pcd", help="Output path for the ground truth map.")
    return parser.parse_args()


def main():
    print("Num o3d device:",o3d.core.cuda.device_count())
    
    args = arg_parse()

    # Ensure all files and directories exist
    if not os.path.exists(args.pose):
        raise FileNotFoundError(f"Pose file does not exist: {args.pose}")
    if not os.path.isdir(args.velodyne):
        raise NotADirectoryError(f"Velodyne directory does not exist: {args.velodyne}")
    if not os.path.exists(args.calib):
        raise FileNotFoundError(f"Calibration file does not exist: {args.calib}")

    # Load calibration and poses
    T_cam2velo = load_calibration(args.calib)
    poses = load_poses(args.pose)
    poses_velo = [T_cam2velo @ pose for pose in poses]

    # Initialize an Open3D point cloud for the global map
    global_map = o3d.geometry.PointCloud()

    # Process each LiDAR scan
    datas = tqdm(enumerate(poses_velo), total=len(poses_velo))
    
    for i, pose in datas:
        scan_path = os.path.join(args.velodyne, f"{i:06d}.bin")
        if not os.path.exists(scan_path):
            print(f"Warning: LiDAR scan file missing: {scan_path}")
            continue

        # Load LiDAR scan
        points = load_lidar_scan(scan_path)

        # Transform points to the global frame
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack((points, ones))  # Convert to homogeneous coordinates
        points_global = (pose @ points_homogeneous.T).T[:, :3]  # Apply pose transformation
        
        # Extend the global map
        global_map.points.extend(o3d.utility.Vector3dVector(points_global))

    print("map constructed!")        

    # Save the global map
    o3d.io.write_point_cloud(args.output, global_map)
    print(f"Ground Truth LiDAR Map Saved as '{args.output}'")


    visualize_point_cloud(np.asarray(global_map.points))



if __name__ == '__main__':
    main()
