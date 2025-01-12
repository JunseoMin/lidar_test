import os
import numpy as np
from glob import glob

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def generate_sparse_point_cloud(points, sparse_factor=4, vertical_angle_bins=64):
    distances = np.linalg.norm(points[:, :3], axis=1)
    vertical_angles = np.arcsin(points[:, 2] / distances)
    vertical_angles = np.degrees(vertical_angles)

    min_angle, max_angle = vertical_angles.min(), vertical_angles.max()
    bin_edges = np.linspace(min_angle, max_angle, vertical_angle_bins + 1)
    bin_indices = np.digitize(vertical_angles, bin_edges) - 1

    selected_points = []
    for i in range(0, vertical_angle_bins, sparse_factor):
        mask = (bin_indices == i)
        selected_points.append(points[mask])

    selected_points = np.vstack(selected_points) if selected_points else np.empty((0, 4))
    return selected_points

def save_point_cloud_to_bin(points, output_file_path):
    points = points.astype(np.float32)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    points.tofile(output_file_path)

def process_sequence(sequence_path, output_dir, sparse_factor, seq):
    file_paths = sorted(glob(os.path.join(sequence_path, "*.bin")))

    target_dir = os.path.join(output_dir, os.path.basename(sequence_path))
    os.makedirs(target_dir, exist_ok=True)

    for idx, file_path in enumerate(file_paths):
        raw_points = load_kitti_bin(file_path)
        sparse_points = generate_sparse_point_cloud(raw_points, sparse_factor=sparse_factor)
        output_path = os.path.join(target_dir, f"{seq:02d}/{idx:06d}.bin")
        save_point_cloud_to_bin(sparse_points, output_path)

output_dir = "/home/server01/js_ws/dataset/reconstruction_input/"
sparse_factor = 4
sequence_paths = sorted(glob("/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences/*/velodyne"))

# Assign specific sequences for validation and test
validation_sequence = sequence_paths.pop(0)  # Use the last sequence for validation
test_sequence = sequence_paths.pop(0)       # Use the second-to-last sequence for test

seq = 2
# Process train sequences
for sequence_path in sequence_paths:
    print(f"Processing train sequence: {sequence_path}")
    process_sequence(sequence_path, os.path.join(output_dir, "train"), sparse_factor, seq)
    seq += 1

# Process validation sequence
print(f"Processing validation sequence: {validation_sequence}")
process_sequence(validation_sequence, os.path.join(output_dir, "validation"), sparse_factor, 0)

# Process test sequence
print(f"Processing test sequence: {test_sequence}")
process_sequence(test_sequence, os.path.join(output_dir, "test"), sparse_factor, 1)

print("All sequences processed!")
