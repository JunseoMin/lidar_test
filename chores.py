import torch
import torch.nn as nn

class FinalBlockWithAttention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super().__init__()
        # 1. Linear Transformation: Input -> Hidden
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 2. Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_features, num_heads=num_heads, batch_first=True)
        
        # 3. Linear Transformation: Hidden -> Output
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Activation and Dropout
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # LayerNorm for stability
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)

    def forward(self, x):
        # x shape: (batch_size, num_points, in_features)
        batch_size, num_points, _ = x.size()
        
        # 1. Input Transformation
        x = self.fc1(x)  # Shape: (batch_size, num_points, hidden_features)
        x = self.act(x)
        x = self.dropout(x)
        
        # 2. Multi-Head Attention
        # Attention requires shape (batch_size, seq_length, embed_dim)
        x_residual = x  # Save residual connection
        x, _ = self.attention(x, x, x)  # Self-attention (Q=K=V=x)
        x = self.norm1(x + x_residual)  # Add & Norm
        
        # 3. Output Transformation
        x_residual = x  # Save another residual connection
        x = self.fc2(x)  # Shape: (batch_size, num_points, out_features)
        x = self.norm2(x + x_residual)  # Add & Norm
        
        return x

# FinalBlockWithAttention 초기화
in_features = 32         # 입력 채널
hidden_features = 64     # Attention 차원
out_features = 3         # 출력 채널
num_heads = 4            # Multi-Head Attention 헤드 수
final_block = FinalBlockWithAttention(in_features, hidden_features, out_features, num_heads)

# 입력 데이터 생성: batch_size=4, num_points=3770, in_features=32
x = torch.randn(3770, 32)  # Shape: (batch_size, num_points, in_features)

# 출력 확인
output = final_block(x)
print("Output shape:", output.shape)
