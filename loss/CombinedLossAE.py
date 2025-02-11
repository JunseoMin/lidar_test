import torch
import torch.nn as nn
import teaserpp_python
from geomloss import SamplesLoss
import torch.nn.functional as F

import time

def replace_if_nan_with_gaussian(x, mean=0.0, std=1.0):
    """
    If there is at least one NaN in x, replace the entire x with Gaussian (normal distribution) random values.
    
    Args:
        x (Tensor): Input tensor.
        mean (float): Mean of the generated random values.
        std (float): Standard deviation of the generated random values.
        
    Returns:
        Tensor: Tensor x without NaN values.
    """
    # Check if x contains any NaN values
    if torch.isnan(x).any():
        # Replace the entire x with samples from N(mean, std^2)
        gauss = torch.randn_like(x) * std + mean
        return gauss
    
    return x

class CombinedCriterionAE(nn.Module):
    def __init__(self, alpha = 0.1 ,beta = 0.45, gamma = 0.45, max_scale = 10000, max_trans = 10000, max_time = 1200):
        super(CombinedCriterionAE, self).__init__()
        self.rec_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)
        # self.inlier_loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.95)
        self.max_time = max_time
        self.l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')

        self.rec_loss_val = 0.
        self.inlier_loss_val = 0.
        self.norm_loss_val = 0.
        
    def forward(self, pred_feat, pred_decoder, input_data, gt_data, train_decoder = False):
        r"""
        @param
        pred_feat [N,6] : x,y,z,nx,ny,nz     - encoder output
        pred_decoder [M,4] : x,y,z,i         - Autoencoder decoder output (for reconstruction loss)
        input_data [M,4] : x,y,z,i           - Autoencoder input data
        gt_data [L,6] : x,y,z,nx,ny,nz       - map data

        @brief: Calc rotation/translation loss + normal loss + reconstruction loss
        """
        if train_decoder:
            rec_loss = self.rec_loss(input_data[:,:3], pred_decoder[:,:3])
            rec_loss += self.l1_loss(input_data[:,3], pred_decoder[:,3])    # Calc Intensity 
            self.rec_loss_val = rec_loss

            return rec_loss

        device = pred_feat.device

        pred_points  = pred_feat[:, :3].to(device)
        pred_normals = pred_feat[:, 3:].to(device)
        gt_points    = gt_data[:, :3].to(device)
        gt_normals   = gt_data[:, 3:].to(device)

        if torch.isnan(pred_normals).any():
            print("Output normal has Nan")
        if torch.isnan(pred_points).any():
            print("Output Points has Nan")

        pred_points = replace_if_nan_with_gaussian(pred_points)

        dists = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)  # (N, L)
        min_indices = torch.argmin(dists, dim=1)  

        inlier_loss = F.mse_loss(pred_points, gt_points[min_indices], reduction='mean')

        closest_gt_normals = gt_normals[min_indices]  # (N, 3)

        pred_normals_unit = F.normalize(pred_normals, p=2, dim=1, eps=1e-4)
        gt_normals_unit   = F.normalize(closest_gt_normals, p=2, dim=1, eps=1e-4)

        cos_theta   = torch.sum(pred_normals_unit * gt_normals_unit, dim=1)
        norm_loss   = (1 - cos_theta).mean()
        
        if torch.isnan(norm_loss).any():
            print("[WARN] Norm loss is NaN!!")

        self.inlier_loss_val = inlier_loss
        self.norm_loss_val = norm_loss

        combined_loss = inlier_loss + norm_loss

        return combined_loss

    def set_max_lower(self, max_trans, max_scale):
        self.max_trans = max_trans
        self.max_scale = max_scale

    def get_loss(self):
        r"""
        returns reconstruction loss, regression loss, norm loss
        """
        return self.rec_loss_val, self.inlier_loss_val, self.norm_loss_val
