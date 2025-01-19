import os
import sys
import struct
import math
import concurrent.futures
import numpy as np
import open3d as o3d
from pathlib import Path

def load_bin_file(bin_path):
    """
    Read a .bin file containing float x,y,z (3 floats per point),
    return an open3d.geometry.PointCloud.
    """
    # Read all bytes
    with open(bin_path, 'rb') as f:
        data = f.read()
    file_size = len(data)
    if file_size % 12 != 0:
        print(f"[Warning] {bin_path} size not multiple of 12 => might be corrupted or 4D data.")
    num_points = file_size // 12

    # Unpack binary => Nx3 float
    points = np.frombuffer(data, dtype=np.float32, count=num_points*3)
    points = points.reshape((-1, 3))  # shape (N,3)

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def save_bin_file(pcd, output_path):
    """
    Save open3d.geometry.PointCloud to .bin (x,y,z) float
    """
    points = np.asarray(pcd.points, dtype=np.float32)
    # Flatten to 1D bytes
    with open(output_path, 'wb') as f:
        f.write(points.tobytes())

def voxel_downsample(pcd, leaf_size):
    """
    Voxel downsample with open3d
    """
    return pcd.voxel_down_sample(voxel_size=leaf_size)

def process_one_file(bin_path, input_dir, output_dir, leaf_size):
    """
    Load -> voxel downsample -> save
    Keep relative path structure
    """
    bin_path = Path(bin_path)
    relative_path = bin_path.relative_to(input_dir)
    out_path = Path(output_dir) / relative_path

    # Create parent dirs
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    pcd = load_bin_file(str(bin_path))
    if len(pcd.points) == 0:
        print(f"[Warning] skip empty or failed: {bin_path}")
        save_bin_file(pcd, out_path)
        print(f"[Done] {bin_path} -> {out_path} ({len(pcd.points)} pts)")
        return

    # Downsample
    ds_pcd = voxel_downsample(pcd, leaf_size)

    # Save
    save_bin_file(ds_pcd, str(out_path))
    print(f"[Done] {bin_path} -> {out_path} ({len(ds_pcd.points)} pts)")

def main():
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} <input_dir> <output_dir> <leaf_size>")
        return

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]
    leaf_size  = float(sys.argv[3])

    print(f"[Info] input_dir  = {input_dir}")
    print(f"[Info] output_dir = {output_dir}")
    print(f"[Info] leaf_size  = {leaf_size}")

    bin_files = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".bin"):
                bin_files.append(os.path.join(root, fname))

    # Sort
    bin_files.sort()

    print(f"[Info] Found {len(bin_files)} bin files.")
    
    # Option 1) Single-thread
    # for path in bin_files:
    #     process_one_file(path, input_dir, output_dir, leaf_size)

    # Option 2) Multi-thread with concurrent futures
    max_workers = min(16, os.cpu_count() or 1)  # up to 8 or #cpu
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in bin_files:
            futures.append(executor.submit(process_one_file, path, input_dir, output_dir, leaf_size))
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()

    print("[Info] All done.")

if __name__ == '__main__':
    main()
