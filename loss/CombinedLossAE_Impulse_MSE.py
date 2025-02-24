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

class CombinedCriterionAEImpulse(nn.Module):
    def __init__(self, alpha = 0.1 ,beta = 0.45, gamma = 0.45, max_scale = 10000, max_trans = 10000, max_time = 1200):
        super(CombinedCriterionAEImpulse, self).__init__()
        self.rec_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)
        self.inlier_loss_var = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.95)
        self.inlier_loss = nn.MSELoss()
        self.max_time = max_time

        self.l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')


        self.rec_loss_val = 0.
        self.inlier_loss_val = 0.
        self.norm_loss_val = 0.

    def repulsion_loss(self, pred_points, alpha = 100.0, margin=0.3):
        """
        Penalize predicted points if they get too close to each other.
        Avoid in-place ops for proper gradient flow.

        pred_points: (N, D)
        margin: minimum desired distance between points
        Returns:
          Scalar (Tensor) representing the repulsion penalty
        """
        N = pred_points.shape[0]
        if N < 2:
            return pred_points.new_zeros(1)  # No repulsion needed if only one or zero points

        # 1) Pairwise distances among predicted points
        # This is differentiable
        dist_matrix = torch.cdist(pred_points, pred_points, p=2)  # shape (N, N)

        # 2) Create a mask or add a large constant to the diagonal
        # Instead of fill_diagonal_(∞), let's do an additive approach.
        # Make an identity matrix with the same size as dist_matrix
        eye = torch.eye(N, device=dist_matrix.device, dtype=dist_matrix.dtype)
        # Add a very large value (e.g., 1e6) to the diagonal so min distance won't pick self-distance
        # This addition is out-of-place, creating a new Tensor without modifying dist_matrix in-place.
        dist_no_self = dist_matrix + 1e6 * eye

        # 3) For each point, find distance to its closest other point
        # so we can enforce margin > min_dist_to_other
        min_dist_to_other, _ = dist_no_self.min(dim=1)  # shape (N,)

        # 4) Hinge-like penalty: if min_dist < margin, penalize (margin - min_dist)^2
        penalty = torch.nn.functional.softplus(alpha * (margin - min_dist_to_other))
        repulsion = (penalty ** 2).mean()

        return repulsion
    
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
        
        # var_loss = self.inlier_loss_var(pred_points, gt_points)
        var_loss = 0.
        if torch.isnan(pred_normals).any():
            print("Output normal has Nan")
        if torch.isnan(pred_points).any():
            print("Output Points has Nan")

        pred_points = replace_if_nan_with_gaussian(pred_points)

        dists = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)  # (N, L)
        min_indices = torch.argmin(dists, dim=1)  

        closest_gt_points = gt_points[min_indices]  # (N, 3)
    
        attraction_loss = F.mse_loss(pred_points, closest_gt_points, reduction='mean')
        # pred_points_copy = pred_points.clone()

        repulsion_loss = self.repulsion_loss(pred_points=pred_points)

        # print("repulsion: ", repulsion_loss)
        # print("attraction: ", attraction_loss)
        inlier_loss = attraction_loss + repulsion_loss
        ## ----- regression loss
        
        closest_gt_normals = gt_normals[min_indices]  # (N, 3)

        pred_normals_unit = F.normalize(pred_normals, p=2, dim=1, eps=1e-5)
        gt_normals_unit   = F.normalize(closest_gt_normals, p=2, dim=1, eps=1e-5)

        cos_theta   = torch.sum(pred_normals_unit * gt_normals_unit, dim=1)
        norm_loss   = (1 - cos_theta).mean()
        
        if torch.isnan(norm_loss).any():
            print("[WARN] Norm loss is NaN!!")

        self.inlier_loss_val = inlier_loss
        self.norm_loss_val = norm_loss

        combined_loss = inlier_loss + norm_loss * 10.0 + var_loss
        # print(combined_loss)
        return combined_loss

    def set_max_lower(self, max_trans, max_scale):
        self.max_trans = max_trans
        self.max_scale = max_scale

    def get_loss(self):
        r"""
        returns reconstruction loss, regression loss, norm loss
        """
        return self.rec_loss_val, self.inlier_loss_val, self.norm_loss_val
