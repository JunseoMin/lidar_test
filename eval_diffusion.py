from model import *
from util import *
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

device_eval = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LiDARDiffusion(
    condition_drop_path = 0.3, 
    condition_enc_block_depth = (1, 1, 2), 
    condition_enc_channels = (8, 16, 32), 
    condition_enc_n_heads = (2, 4, 4),
    condition_enc_patch_size = (1024, 1024, 1024), 
    condition_qkv_bias = True, 
    condition_qk_scale = None, 
    condition_attn_drop  = 0.1, 
    condition_proj_drop = 0.1, 
    condition_mlp_ratio = 4, 
    condition_stride = (2, 2, 2), 
    condition_in_channels = 4,
    condition_out_channel = 3,
    condition_hidden_channel = 8,
    drop_path = 0.3,
    enc_block_depth = (2, 2, 2, 3, 2),
    enc_channels = (16, 32, 64, 128, 256),
    enc_n_heads = (2, 2, 4, 4, 16),
    enc_patch_size = (1024, 1024, 1024, 1024, 1024), 
    qkv_bias = True, 
    qk_scale = None, 
    attn_drop = 0.1, 
    proj_drop = 0.1, 
    mlp_ratio = 4, 
    stride = (2, 2, 2, 2),
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    dec_depths = (2, 2, 2, 2),
    dec_channels = (16, 32, 64, 128),
    dec_n_head = (2, 4, 8, 8),
    dec_patch_size = (1024, 1024, 1024, 1024),
    time_out_ch = 3,
    num_steps = 5000,
    beta_1 = 1e-6,
    beta_T = 1e-2,
    device = device_eval
)

ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/best_model_reconst.pth"
checkpoint = torch.load(ckpt_dir, map_location="cuda", weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device_eval)
model.eval()

gt = load_kitti_bin_gt("/home/server01/js_ws/dataset/reconstruction_dataset/reconst_gt/test/01/000000.bin")
test = kitti_to_dict("/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/test/velodyne/01/000000.bin", device_eval)

start_time = time.time()
reconst = model.sample(test["feat"].shape[0], test, device_eval, is_validation=False)
print(f'processing time: {time.time() - start_time}')
point_to_bin(reconst[2000],"/home/server01/js_ws/lidar_test/evaluate_output/diffusion/000000-2000.bin")
point_to_bin(reconst[1000],"/home/server01/js_ws/lidar_test/evaluate_output/diffusion/000000-1000.bin")
point_to_bin(reconst[0],"/home/server01/js_ws/lidar_test/evaluate_output/diffusion/000000-0.bin")