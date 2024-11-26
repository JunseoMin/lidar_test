import torch
import torch.nn as nn

class ExpandMLP(nn.Module):
    def __init__(self, input_dim, output_dim, expand_factor):
        super().__init__()
        self.expand_factor = expand_factor  # 30
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * expand_factor),  # C -> 30 * C/30
            nn.ReLU()
        )

    def forward(self, x):
        # x의 크기: (N, C)
        expanded = self.mlp(x)  # (N, 30 * C/30)
        expanded = expanded.view(-1, self.expand_factor, expanded.size(-1) // self.expand_factor)
        # 크기 변환: (N, 30, C/30)
        expanded = expanded.view(-1, expanded.size(-1))  # (30N, C/30)
        return expanded

# 예제: 입력 및 출력
N, C = 2, 64
expand_factor = 32
input_feat = torch.rand(N, C)  # 입력 텐서 (N, C)

# MLP 초기화
mlp = ExpandMLP(input_dim=C, output_dim=C // expand_factor, expand_factor=expand_factor)

# 출력
output_feat = mlp(input_feat)
print("출력 크기:", output_feat.shape)  # (30N, C/30)
