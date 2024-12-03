import torch
import numpy as np
from glob import glob
import random
import os
from multiprocessing import Pool
from sklearn.cluster import DBSCAN

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def generate_sparse_point_cloud(points, sparse_factor=4, vertical_angle_bins=64):
    """
    Filters the point cloud data by vertical angles based on the inclination angle (z-axis direction),
    and selects points in specific angular bins based on sparse_factor.

    Args:
        points (np.ndarray): Original point cloud data of shape (N, 4), where N is the number of points.
                             Each point is represented as [x, y, z, intensity].
        sparse_factor (int): Factor to select vertical angular regions. Points within every nth vertical angular bin are selected.
        vertical_angle_bins (int): Total number of vertical angular bins (default: 64 for a 64-channel LiDAR).

    Returns:
        np.ndarray: Downsampled point cloud data of shape (M, 4), where M <= N.
    """
    # Step 1: Calculate vertical angles (inclination angles)
    distances = np.linalg.norm(points[:, :3], axis=1)  # Distance from the origin
    vertical_angles = np.arcsin(points[:, 2] / distances)  # Calculate inclination angle in radians
    vertical_angles = np.degrees(vertical_angles)  # Convert to degrees

    # Step 2: Divide the vertical angle range into bins
    min_angle, max_angle = vertical_angles.min(), vertical_angles.max()  # Define vertical angle range
    bin_edges = np.linspace(min_angle, max_angle, vertical_angle_bins + 1)  # Define vertical bins
    bin_indices = np.digitize(vertical_angles, bin_edges) - 1  # Determine bin index for each point

    # Step 3: Select points from specific angular bins based on sparse_factor
    selected_points = []
    for i in range(0, vertical_angle_bins, sparse_factor):
        mask = (bin_indices == i)  # Select points in the current vertical bin
        selected_points.append(points[mask])

    # Step 4: Combine all selected points
    selected_points = np.vstack(selected_points) if selected_points else np.empty((0, 4))
    
    return selected_points

def save_point_cloud_to_bin(points, output_file_path):
    points = points.astype(np.float32)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    points.tofile(output_file_path)


def generate_unique_filename(file_path, output_dir, idx):
    """
    Generate a unique filename by hashing the file path.
    :param file_path: Original file path
    :param output_dir: Output directory
    :return: Unique file path
    """
    base_name = os.path.basename(file_path)
    unique_name = f"{idx}.bin"
    return os.path.join(output_dir, unique_name)


def process_file(file_path, input_dir, gt_dir, sparse_factor, idx):
    raw_points = load_kitti_bin(file_path)
    # raw_points = point["feat"].numpy()

    gt_output_path = generate_unique_filename(file_path, gt_dir, idx)
    save_point_cloud_to_bin(raw_points, gt_output_path)

    sparse_points = generate_sparse_point_cloud(raw_points, sparse_factor=sparse_factor)
    train_output_path = generate_unique_filename(file_path, input_dir, idx)
    save_point_cloud_to_bin(sparse_points, train_output_path)

    print(f"Processed: {file_path} -> GT: {gt_output_path}, Sparse: {train_output_path}")

def process(file_paths, train_dir, gt_dir, test_dir, gt_test_dir, validation_dir, gt_validation_dir, sparse_factor, num_train, num_test, num_validation):
    random.shuffle(file_paths)

    train_file_paths = file_paths[:num_train]
    test_file_paths = file_paths[num_train:num_train+num_test]
    validation_file_paths = file_paths[num_train+num_test:]
    
    # process train data
    for i in range(len(train_file_paths)):
        process_file(file_path=train_file_paths[i], input_dir=train_dir, gt_dir= gt_dir, sparse_factor=sparse_factor, idx=i)

    # process test data
    for i in range(len(test_file_paths)):
        process_file(file_path=test_file_paths[i], input_dir=test_dir, gt_dir= gt_test_dir, sparse_factor=sparse_factor, idx=i)

    for i in range(len(validation_file_paths)):
        process_file(file_path=validation_file_paths[i], input_dir=validation_dir, gt_dir= gt_validation_dir, sparse_factor=sparse_factor, idx=i)
    return len(file_paths)

dataset_path = glob("/home/server01/js_ws/dataset/**/*.bin", recursive=True)

# train dataset
train_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/train"
gt_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/train_GT"

# test dataset
test_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/test"
gt_test_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/test_GT"

# validation dataset
validation_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/validation"
gt_validation_output_dir = "/home/server01/js_ws/dataset/vertical_downsampled/validation_GT"

sparse_factor = 4
num_train = 24000
num_test = 2400
num_validation = 1000

dataset_len = process(dataset_path,train_output_dir,gt_output_dir, test_output_dir, gt_test_output_dir, validation_output_dir, gt_validation_output_dir, sparse_factor, num_train, num_test, num_validation)

print(f"process finished! dataset data: {dataset_len}")