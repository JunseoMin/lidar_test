import torch
import numpy as np
from glob import glob
from model.PTv3 import Point
import random

# KITTI .bin 파일을 로드하는 함수
def load_kitti_bin(file_path):
    """
    Load a KITTI .bin file as a numpy array.
    :param file_path: Path to .bin file
    :return: Numpy array of shape (N, 4) [x, y, z, intensity]
    """
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# KITTI .bin 파일을 Point 객체로 변환하는 함수
def kitti_to_point(file_path, grid_size=0.05, batch_id=0):
    """
    Convert a KITTI .bin file to a Point object suitable for the model.
    :param file_path: Path to .bin file
    :param grid_size: Grid size for quantization
    :param batch_id: Batch ID for the point cloud
    :return: Point object
    """
    # KITTI 점 클라우드 로드
    raw_data = load_kitti_bin(file_path)
    
    # 좌표(x, y, z)와 특성(intensity) 분리
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity
    
    # 좌표와 intensity를 결합하여 features 생성
    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),  # 좌표 (x, y, z)
        torch.tensor(intensity, dtype=torch.float32)  # intensity
    ], dim=1)
    
    # 배치 ID 설정
    batch_tensor = torch.full((features.shape[0],), batch_id, dtype=torch.int64)
    
    # Point 객체 생성
    point = Point({
        "coord": features[:, :3],  # 원본 좌표 (x, y, z)
        "feat": features,          # 특성 (x, y, z, intensity)
        "batch": batch_tensor,     # 배치 ID
        "grid_size": torch.tensor([grid_size]),  # 양자화(grid size)
    })
    return point

# 희소한 입력 데이터셋 생성
def generate_sparse_point_cloud(points, scale_factor=0.5):
    """
    Create a sparse point cloud by scaling down the input points randomly.
    :param points: Original point cloud
    :param scale_factor: Scale factor to reduce the density of the point cloud
    :return: Sparse point cloud
    """
    # 원본 점 클라우드에서 일부 포인트를 랜덤하게 선택하여 희소한 포인트 클라우드 생성
    num_points = points.shape[0]
    num_sparse_points = int(num_points * scale_factor)
    
    # 랜덤하게 포인트 샘플링
    indices = random.sample(range(num_points), num_sparse_points)
    sparse_points = points[indices, :]
    
    return sparse_points

# KITTI .bin 파일 경로 설정
file_paths = glob("/home/junseo/datasets/kitti/2011_09_28/2011_09_28_drive_0002_sync/velodyne_points/data/*.bin", recursive=True)

# 입력 데이터셋 생성
input_points_list = []
for i, fp in enumerate(file_paths):
    point = kitti_to_point(fp, grid_size=0.05, batch_id=i)
    
    # 희소한 점 클라우드 생성 (랜덤 스케일 적용)
    sparse_points = generate_sparse_point_cloud(point["feat"].numpy(), scale_factor=0.5)
    sparse_point_tensor = torch.tensor(sparse_points, dtype=torch.float32)
    
    input_points_list.append(sparse_point_tensor)

# 출력 데이터셋 생성
# 원본 LiDAR 데이터(64채널)를 출력 데이터로 사용
output_points_list = []
for i, fp in enumerate(file_paths):
    raw_data = load_kitti_bin(fp)
    
    # 출력은 원본 데이터 그대로 (64채널 LiDAR 데이터)
    output_points_list.append(torch.tensor(raw_data, dtype=torch.float32))

# 데이터셋 생성 완료 후, 모델 학습을 위한 예시
print("Input Point Dataset:", len(input_points_list), "samples")
print("Output Point Dataset:", len(output_points_list), "samples")
