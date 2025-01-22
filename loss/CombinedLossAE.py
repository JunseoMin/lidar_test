import torch
import torch.nn as nn
import teaserpp_python
from geomloss import SamplesLoss
import torch.nn.functional as F

class CombinedCriterionAE(nn.Module):
    def __init__(self, alpha = 0.1 ,beta = 0.45, gamma = 0.45):
        super(CombinedCriterionAE, self).__init__()
        self.rec_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)
        
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        self.solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred_feat, pred_decoder, input_data, gt_data):
        r"""
        pred_feat [N,6] : x,y,z,nx,ny,nz     - encoder output
        pred_decoder [M,4] : x,y,z,i         - Autoencoder decoder output (for reconstruction loss)
        input_data [M,4] : x,y,z,i           - Autoencoder input data
        gt_data [L,6] : x,y,z,nx,ny,nz       - map data

        @brief: Calc rotation/translation loss + normal loss + reconstruction loss
        """
        device = pred_feat.device
        rec_loss = self.rec_loss(input_data, pred_decoder)

        pred_feat_np = pred_feat[:, :3].detach().cpu().numpy()  # xyz만
        gt_data_np = gt_data[:, :3].detach().cpu().numpy()      # xyz만

        self.solver.solve(pred_feat_np, gt_data_np)
        solution = self.solver.getSolution()

        R = solution.rotation  # shape (3,3)
        t = solution.translation  # shape (3,)
        s = solution.scale       # float
        
        R_torch = torch.from_numpy(R).float().to(device)
        t_torch = torch.from_numpy(t).float().to(device)
        s_torch = torch.tensor(s).float().to(device)

        I = torch.eye(3).to(device)
        rot_loss   = torch.norm(R_torch - I, p='fro')    
        trans_loss = torch.norm(t_torch, p=2)            
        scale_loss = (s_torch - 1) ** 2                  

        reg_loss = rot_loss + trans_loss + scale_loss

        pred_points  = pred_feat[:, :3].to(device)
        pred_normals = pred_feat[:, 3:].to(device)
        gt_points    = gt_data[:, :3].to(device)
        gt_normals   = gt_data[:, 3:].to(device)

        dists = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)  # (N, L)
        min_indices = torch.argmin(dists, dim=1)  

        closest_gt_normals = gt_normals[min_indices]  # (N, 3)

        pred_normals_unit = F.normalize(pred_normals, p=2, dim=1)
        gt_normals_unit   = F.normalize(closest_gt_normals, p=2, dim=1)

        cos_theta   = torch.sum(pred_normals_unit * gt_normals_unit, dim=1)
        norm_loss   = (1 - cos_theta).mean()
        
        self.rec_loss = rec_loss
        self.reg_loss = reg_loss
        self.norm_loss = norm_loss

        combined_loss = self.alpha * rec_loss + self.beta * reg_loss + self.gamma * norm_loss
        return combined_loss
    

    def get_loss(self):
        r"""
        returns reconstruction loss, regression loss, norm loss
        """
        return self.rec_loss, self.reg_loss, self.norm_loss

    def set_weights(self, alpha,beta,gamma):
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO
        pass
