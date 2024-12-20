import numpy as np
import open3d as o3d
import os
import argparse
from tqdm import tqdm

def load_calibration(calib_file):
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
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"LiDAR scan file not found: {scan_path}")
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]  # x, y, z (intensity 제외)
    return points

def load_label(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    label_data = np.fromfile(label_path, dtype=np.uint32)
    semantic_label = label_data & 0xFFFF
    return semantic_label

def arg_parse():
    parser = argparse.ArgumentParser(description="Generate a semantic segmentation map from KITTI dataset.")
    parser.add_argument('--pose', type=str, default="/home/junseo/datasets/kitti_odometry/gt_pose/dataset/poses/09.txt", help="Path to KITTI pose file.")
    parser.add_argument('--velodyne', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_velodyne/dataset/sequences/09/velodyne", help="Path to KITTI LiDAR scans folder.")
    parser.add_argument('--labels', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_labels/dataset/sequences/09/labels", help="Path to KITTI Label folder.")
    parser.add_argument('--calib', type=str, default="/home/junseo/datasets/kitti_odometry/data_odometry_calib/dataset/sequences/09/calib.txt", help="Path to calibration file.")
    parser.add_argument('--output', type=str, default="/home/junseo/MPIL/implementations/lidar_sr/map/09.bin", help="Output path for the semantic map .bin file.")
    return parser.parse_args()

def main():
    args = arg_parse()
    # Calibration
    T_cam2velo = load_calibration(args.calib)  
    # Pose
    poses = load_poses(args.pose)

    # 파일 리스트
    scan_files = sorted(os.listdir(args.velodyne))
    
    # 전체 맵을 담을 리스트
    global_points = []
    global_labels = []

    datas = tqdm(enumerate(scan_files), total=len(scan_files))
    for i, scan_file in datas:
        scan_path = os.path.join(args.velodyne, scan_file)
        label_file = scan_file.replace('.bin', '.label')
        label_path = os.path.join(args.labels, label_file)

        if not os.path.exists(scan_path):
            print(f"Warning: LiDAR scan file missing: {scan_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Warning: Label file missing: {label_path}")
            continue

        points = load_lidar_scan(scan_path)
        semantic_label = load_label(label_path)

        # Homogeneous
        ones = np.ones((points.shape[0], 1))
        points_hom = np.hstack((points, ones))

        # Velodyne->Camera->Global
        points_camera = (T_cam2velo @ points_hom.T).T
        points_global = (poses[i] @ points_camera.T).T[:, :3]

        global_points.append(points_global)
        global_labels.append(semantic_label)

    # 모든 프레임 포인트를 하나로 합침
    global_points = np.vstack(global_points)
    global_labels = np.concatenate(global_labels)

    # [x, y, z, label]
    # x,y,z는 float32, label은 int32로 저장
    output_data = np.hstack((global_points.astype(np.float32), global_labels[:, None].astype(np.int32)))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data.tofile(args.output)
    print(f"Semantic Map Saved as '{args.output}' with shape {output_data.shape}")

    # 시각화
    # label -> color 매핑을 위한 colormap 예시 (단순 랜덤 또는 규정된 colormap 사용 가능)
    unique_labels = np.unique(global_labels)
    # 임의의 랜덤 컬러 맵 (고정 시드)
    np.random.seed(42)
    max_label = unique_labels.max()
    colormap = np.random.rand(max_label+1, 3)

    colors = colormap[global_labels]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Open3D로 시각화
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
