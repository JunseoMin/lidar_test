from model.LidarEncoder import PTEncoder
import torch
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PTEncoder(
                 in_channels = 4,
                 drop_path = 0.3,
                 enc_depths = (2, 2, 4, 6, 4, 2, 2),
                 enc_channels = (32, 64, 128, 256, 512, 256, 128),
                 enc_num_head = (2, 4, 8, 16, 32, 16, 8),
                 enc_patch_size = (1024, 1024, 1024, 1024, 1024, 1024, 1024), 
                 qkv_bias = True, 
                 qk_scale = None, 
                 attn_drop = 0.1, 
                 proj_drop = 0.1, 
                 mlp_ratio = 4, 
                 stride = (2, 2, 4, 4, 4, 2),
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 out_channels=6,
                 dec_depths=(1, 1, 2, 2, 1, 1),
                 dec_channels=(4, 8, 16, 32, 64, 64),
                 dec_num_head=(2, 2, 4, 8, 8, 8),
                 dec_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
).to(device=device)

model_path = "/home/server01/js_ws/lidar_test/ckpt/best_model_encoding.pth"
test_path = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/validation/velodyne/00/000000.bin"

input = kitti_to_dict(test_path, device=device)

checkpoint = torch.load(model_path, map_location="cuda", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device=device)
model.eval()

with torch.no_grad():
    pred, _ = model(input, False)
    point_to_bin(pred,"/home/server01/js_ws/lidar_test/evaluate_output/encoder_output/000000.bin")
    