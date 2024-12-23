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
    parser = argparse.ArgumentParser(description="Generate voxelized semantic maps from KITTI dataset in chunks.")
    parser.add_argument('--dataset_path', type=str, default="/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences",
                        help="Path to KITTI dataset sequences folder")
    parser.add_argument('--output_path', type=str, default="/home/server01/js_ws/dataset/odometry_dataset/GT_map",
                        help="Output path for semantic maps.")
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help="Voxel size for down-sampling.")
    parser.add_argument('--chunk_frames', type=int, default=100,
                        help="Number of frames to accumulate before voxelizing.")
    return parser.parse_args()

def voxel_downsample_with_label(points_np, labels_np, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    ds_points = np.asarray(pcd_ds.points)
    if ds_points.shape[0] == 0:
        return ds_points, np.array([], dtype=np.float32)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    ds_labels = np.zeros((ds_points.shape[0],), dtype=np.float32)
    for i, pt in enumerate(ds_points):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
        nearest_index = idx[0]
        ds_labels[i] = labels_np[nearest_index]
    return ds_points, ds_labels

def process_sequence(sequence_id, args):
    sequence_path = os.path.join(args.dataset_path, f"{sequence_id:02d}")
    velodyne_path = os.path.join(sequence_path, "velodyne")
    labels_path = os.path.join(sequence_path, "labels")
    calib_file = os.path.join(sequence_path, "calib.txt")
    poses_file = os.path.join(sequence_path, "poses.txt")
    
    if not os.path.exists(velodyne_path) or not os.path.exists(labels_path):
        print(f"Skipping sequence {sequence_id:02d}: Missing data folder.")
        return None

    T_cam2velo = load_calibration(calib_file)
    poses = load_poses(poses_file)

    scan_files = sorted(os.listdir(velodyne_path))
    global_points_list = []
    global_labels_list = []

    chunk_points = []
    chunk_labels = []

    datas = tqdm(enumerate(scan_files), total=len(scan_files), desc=f"Sequence {sequence_id:02d}")
    for i, scan_file in datas:
        scan_path = os.path.join(velodyne_path, scan_file)
        label_file = scan_file.replace('.bin', '.label')
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(scan_path) or not os.path.exists(label_path):
            continue

        points = load_lidar_scan(scan_path)
        semantic_label = load_label(label_path)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_hom = np.hstack((points, ones))

        points_camera = (T_cam2velo @ points_hom.T).T
        points_global = (poses[i] @ points_camera.T).T[:, :3]

        chunk_points.append(points_global)
        chunk_labels.append(semantic_label)

        if (i + 1) % args.chunk_frames == 0 or (i + 1) == len(scan_files):
            chunk_points_np = np.vstack(chunk_points)
            chunk_labels_np = np.concatenate(chunk_labels)

            ds_points, ds_labels = voxel_downsample_with_label(
                chunk_points_np, chunk_labels_np, args.voxel_size
            )

            global_points_list.append(ds_points)
            global_labels_list.append(ds_labels)

            chunk_points = []
            chunk_labels = []

    global_points = np.vstack(global_points_list) if global_points_list else np.empty((0, 3), dtype=np.float32)
    global_labels = np.concatenate(global_labels_list) if global_labels_list else np.empty((0,), dtype=np.float32)

    output_data = np.hstack([
        global_points.astype(np.float32),
        global_labels.reshape(-1, 1).astype(np.float32)
    ])

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{sequence_id:02d}_map.bin")
    output_data.tofile(output_file)
    print(f"[Done] Semantic Map for Sequence {sequence_id:02d} saved to {output_file}")


def main():
    args = arg_parse()
    for sequence_id in range(11):
        process_sequence(sequence_id, args)

if __name__ == '__main__':
    main()
