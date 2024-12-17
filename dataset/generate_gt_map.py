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
            if line.startswith("Tr:"):
                values = list(map(float, line.strip().split()[1:]))
                T_cam2velo = np.eye(4)
                T_cam2velo[:3, :4] = np.array(values).reshape(3, 4)
                return T_cam2velo
    raise RuntimeError("Calibration matrix 'Tr' not found in the file!")


def load_poses(file_path):
    """Load KITTI ground truth poses."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pose file not found: {file_path}")
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.eye(4)
            values = list(map(float, line.strip().split()))
            T[:3, :4] = np.array(values).reshape(3, 4)
            poses.append(T)
    return poses


def load_lidar_scan(scan_path):
    """Load a single KITTI LiDAR scan."""
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"LiDAR scan file not found: {scan_path}")
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]  # x, y, z (ignore intensity)
    return points

def arg_parse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a ground truth LiDAR map from KITTI dataset.")
    parser.add_argument('--pose', type=str, default="/home/junseo/datasets/kitti_odometry/gt_pose/dataset/poses/08.txt", help="Path to KITTI pose file.")
    parser.add_argument('--velodyne', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_velodyne/dataset/sequences/08/velodyne", help="Path to KITTI LiDAR scans folder.")
    parser.add_argument('--calib', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_calib/dataset/sequences/08/calib.txt", help="Path to calibration file.")
    parser.add_argument('--output', type=str, default="/home/junseo/MPIL/implementations/lidar_sr/map/08.pcd", help="Output path for the ground truth map.")
    return parser.parse_args()


def main():
    args = arg_parse()

    # Load calibration and invert Tr for Velodyne -> Camera
    T_cam2velo = load_calibration(args.calib)  
    
    # Load ground truth poses (Camera)
    poses = load_poses(args.pose)

    # Initialize global point cloud
    global_map = o3d.geometry.PointCloud()

    # Process each LiDAR scan
    scan_files = sorted(os.listdir(args.velodyne))
    datas = tqdm(enumerate(scan_files), total=len(scan_files))

    for i, scan_file in datas:
        scan_path = os.path.join(args.velodyne, scan_file)
        if not os.path.exists(scan_path):
            print(f"Warning: LiDAR scan file missing: {scan_path}")
            continue

        # Load LiDAR scan
        points = load_lidar_scan(scan_path)
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack((points, ones))  # Homogeneous coordinates

        # Velodyne -> Camera -> Global
        points_camera = (T_cam2velo @ points_homogeneous.T).T  # Velodyne to Camera
        points_global = (poses[i] @ points_camera.T).T[:, :3]  # Camera to Global

        # Update the global map
        global_map.points.extend(o3d.utility.Vector3dVector(points_global))

    # Save the global map
    o3d.io.write_point_cloud(args.output, global_map)
    print(f"Ground Truth LiDAR Map Saved as '{args.output}'")

    # Visualize the map
    # o3d.visualization.draw_geometries([global_map])


if __name__ == '__main__':
    main()
