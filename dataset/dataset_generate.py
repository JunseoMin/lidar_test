import torch
import numpy as np
from glob import glob
from model.PTv3 import Point
import random
import os
from multiprocessing import Pool


def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def generate_sparse_point_cloud(points, sparse_factor=1):
    num_points = points.shape[0]
    num_sparse_points = max(1, num_points // sparse_factor)
    indices = random.sample(range(num_points), num_sparse_points)
    return points[indices, :]


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


def process_file(file_path, train_dir, gt_dir, sparse_factor, idx, grid_size=0.05, batch_id=0):
    raw_points = load_kitti_bin(file_path)
    # raw_points = point["feat"].numpy()

    gt_output_path = generate_unique_filename(file_path, gt_dir, idx)
    save_point_cloud_to_bin(raw_points, gt_output_path)

    sparse_points = generate_sparse_point_cloud(raw_points, sparse_factor=sparse_factor)
    train_output_path = generate_unique_filename(file_path, train_dir, idx)
    save_point_cloud_to_bin(sparse_points, train_output_path)

    print(f"Processed: {file_path} -> GT: {gt_output_path}, Sparse: {train_output_path}")


def process_files_in_parallel(file_paths, train_dir, gt_dir, sparse_factor, num_workers=4):
    args = [(file_paths[i], train_dir, gt_dir, sparse_factor, i ) for i in range(len(file_paths))]
    with Pool(num_workers) as pool:
        pool.starmap(process_file, args)

def process(file_paths, train_dir, gt_dir, sparse_factor):
    for i in range(len(file_paths)):
        process_file(file_path=file_paths[i], train_dir=train_dir, gt_dir= gt_dir, sparse_factor=sparse_factor, idx=i)

    return len(file_paths)

train_file_paths = glob("/home/server01/js_ws/dataset/2011_09_2**/**/*.bin", recursive=True)
test_file_paths_0930 = glob("/home/server01/js_ws/dataset/2011_09_30/**/*.bin", recursive=True)
# test_file_paths_1003 = glob("/home/server01/js_ws/dataset/2011_10_03/**/*.bin", recursive=True)
test_file_paths = test_file_paths_0930


# train_ratio = 1
# num_train = int(len(file_paths) * train_ratio)
# random.shuffle(file_paths)

train_files = train_file_paths
test_files = test_file_paths

train_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti_4/train"
test_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti_4/test"
gt_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti_4/GT"
gt_test_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti_4/GT_test"

sparse_factor = 4
  
trainlen = process(train_files,train_output_dir,gt_output_dir,sparse_factor)
testlen = process(test_files,test_output_dir,gt_test_output_dir,sparse_factor)

print(f"process finished! train data: {trainlen} test data: {testlen}")

# process_files_in_parallel(train_files, train_output_dir, gt_output_dir, sparse_factor, num_workers=20)
# process_files_in_parallel(test_files, test_output_dir, gt_output_dir, sparse_factor, num_workers=20)
