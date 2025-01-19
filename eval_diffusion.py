from model import *
from util import *

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
        enc_block_depth = (2, 2, 2, 4, 2),
        enc_channels = (16, 16, 32, 64, 128),
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
        num_steps = 500,
        beta_1 = 1e-4,
        beta_T = 1e-2,
        device=device_eval
    )


ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/best_model_diffusion.pth"
checkpoint = torch.load(ckpt_dir, map_location="cuda", weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device_eval)
model.eval()

gt = load_kitti_bin_gt("/home/server01/js_ws/dataset/reconstruction_dataset/reconst_gt/test/01/000000.bin")
test = kitti_to_dict("/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/test/velodyne/01/000000.bin")

reconst = model.sample(10000, test, device_eval)
point_to_bin(reconst,"/home/server01/js_ws/lidar_test/evaluate_output/diffusion/000000.bin")