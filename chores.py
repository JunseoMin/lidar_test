import torch

# 데이터의 길이
num_points = 16  # 예: 총 데이터 개수
segments = 3     # 구간 수

# 각 구간의 데이터 개수 계산
points_per_segment = (num_points + segments - 1) // segments  # 올림 처리

# 0, 1, 2 반복 생성
batch_tensor = torch.arange(segments).repeat_interleave(points_per_segment)[:num_points]
batch_tensor = batch_tensor.to(dtype=torch.int64)

print(batch_tensor)
