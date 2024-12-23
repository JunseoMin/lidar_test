import numpy as np
from glob import glob
import os
from tqdm import tqdm

from pruningloss import PruingLoss

def load_kitti_bin(file_path):
    """
    Load point cloud data from a .bin file (KITTI format).
    Each point is [x, y, z, intensity] or [x, y, z, label] etc.
    """
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def load_kitti_bin_gt(file_path):
    """
    Load point cloud data from a .bin file (KITTI format).
    Each point is [x, y, z, intensity, label].
    """
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)

def save_point_cloud_to_bin(points, output_file_path):
    """
    Save point cloud data to a .bin file.
    points can be Nx4 or Nx5 etc. Just ensure float32.
    """
    points = points.astype(np.float32)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    points.tofile(output_file_path)

def load_calibration(calib_file):
    """
    Load KITTI calibration file (Tr: = velodyne->camera transform).
    """
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith("Tr:"):
                values = list(map(float, line.strip().split()[1:]))
                T_velo2cam = np.eye(4)
                T_velo2cam[:3, :4] = np.array(values).reshape(3, 4)
                return T_velo2cam
    raise RuntimeError("Calibration matrix 'Tr' not found in the file!")

def load_poses(file_path):
    """
    Load camera->global (KITTI ground truth) poses.
    """
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

def transform_global_to_velo(points_global, pose_cam2global, T_velo2cam):
    """
    Transform points from Global coordinates -> Velodyne coordinates.

    Args:
        points_global: (N,4) or (N,3) array of global points
        pose_cam2global: 4x4 matrix that transforms camera->global
        T_velo2cam: 4x4 matrix that transforms velo->camera (from KITTI calib)

    Returns:
        points_velo: (N,3) Velodyne coordinates
    """
    # 만약 들어온 points_global가 (N,3)라면, Homogeneous 좌표로 만든다.
    if points_global.shape[1] == 3:
        ones = np.ones((points_global.shape[0], 1), dtype=np.float32)
        points_global_hom = np.hstack((points_global, ones))
    else:
        points_global_hom = points_global

    # pose_cam2global의 역행렬: Global->Camera
    pose_global2cam = np.linalg.inv(pose_cam2global)
    # T_velo2cam의 역행렬: Camera->Velodyne
    T_cam2velo = np.linalg.inv(T_velo2cam)

    # Global -> Camera
    points_camera = (pose_global2cam @ points_global_hom.T).T  # (N,4)
    # Camera -> Velodyne
    points_velo = (T_cam2velo @ points_camera.T).T  # (N,4)

    # 결과에서 x,y,z만 사용
    return points_velo[:, :3]

def main():
    path_to_64_query = "/home/server01/js_ws/dataset/odometry_dataset/gt/"
    path_to_map = "/home/server01/js_ws/dataset/odometry_dataset/GT_map/"
    output_path = "/home/server01/js_ws/dataset/odometry_dataset/gt_points/"

    pruing = PruingLoss()

    for seq in range(4, 5):
        # 예: /home/server01/js_ws/dataset/odometry_dataset/gt/04/*.bin
        lidar_paths = glob(path_to_64_query + f"{seq:02d}/*.bin")
        # 예: /home/server01/js_ws/dataset/odometry_dataset/gt_map/04_map.bin
        map_path = path_to_map + f"{seq:02d}_map.bin"

        # 예: .../sequences/04/calib.txt
        calib_path = f"/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences/{seq:02d}/calib.txt"
        # 예: .../sequences/04/poses.txt
        poses_path = f"/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences/{seq:02d}/poses.txt"

        # Load transforms
        T_velo2cam = load_calibration(calib_path)   # Actually velo->cam
        poses_cam2global = load_poses(poses_path)   # camera->global (list of length #frames)

        # Load map, set into pruing
        # map_file: [x, y, z, label], global coords
        print("Loading map:", map_path)
        map_file = load_kitti_bin(map_path)
        pruing.set_map(map_file)
        print(f"sequence {seq:02d} map processed!")

        datas = tqdm(enumerate(lidar_paths), total=len(lidar_paths))
        for i, lidar_path in datas:
            # 1) Load the query point cloud (GT 64-ch in global coords, presumably)
            query = load_kitti_bin_gt(lidar_path)  # shape (N,5): [x, y, z, intensity, label]
            query = query[:, [0, 1, 2, 4]]

            # 2) pruing.get_processed_map()라는 사용자 정의 함수로 처리
            #    (어떤 의미인지는 pruningloss 코드에 따라 다름)
            processed = pruing.get_processed_map(query)
            # 여기서 processed가 (M,4) 등등의 형태라고 가정
            npfile = np.array(processed)  # 안전하게 NumPy array로 변환

            # poses_cam2global[i]: i번째 frame의 Camera->Global pose
            # 따라서 Global->Velodyne 변환을 적용
            transformed = transform_global_to_velo(
                points_global=npfile, 
                pose_cam2global=poses_cam2global[i],
                T_velo2cam=T_velo2cam
            )

            # 저장 경로
            out_dir = os.path.join(output_path, f"{seq:02d}")
            output_file_path = os.path.join(out_dir, f"{i:06d}.bin")

            # 우리가 원하는 최종 포맷이 (N,4)인지 (N,3)인지 확인 후 저장
            # 예: 만약 label이 필요하다면, transformed에 label 붙여서 저장
            # 여기서는 x,y,z만 저장하는 예시
            save_point_cloud_to_bin(transformed, output_file_path)

if __name__ == '__main__':
    main()
