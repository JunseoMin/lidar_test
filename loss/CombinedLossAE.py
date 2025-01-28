import torch
import torch.nn as nn
import teaserpp_python
from geomloss import SamplesLoss
import torch.nn.functional as F

class CombinedCriterionAE(nn.Module):
    def __init__(self, alpha = 0.1 ,beta = 0.45, gamma = 0.45):
        super(CombinedCriterionAE, self).__init__()
        self.rec_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)
        
        self.solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        self.solver_params.cbar2 = 1
        self.solver_params.noise_bound = 0.01
        self.solver_params.estimate_scaling = True
        self.solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        self.solver_params.rotation_gnc_factor = 1.4
        self.solver_params.rotation_max_iterations = 100
        self.solver_params.rotation_cost_threshold = 1e-12

        self.l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.rec_loss_val = 0.
        self.reg_loss_val = 0.
        self.norm_loss_val = 0.
        
    def forward(self, pred_feat, pred_decoder, input_data, gt_data, train_decoder = False):
        r"""
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

        solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)

        device = pred_feat.device
        reg_loss = self.reg_loss_val if hasattr(self, 'reg_loss_val') else torch.tensor(0.0, device=device)

        pred_feat_np = pred_feat[:, :3].detach().cpu().numpy()  # xyz
        gt_data_np = gt_data[:, :3].detach().cpu().numpy()      # xyz
        
        solver.solve(pred_feat_np.T, gt_data_np.T)
        solution = solver.getSolution()
        R = solution.rotation  # shape (3,3)
        t = solution.translation  # shape (3,)
        s = solution.scale       # float

        R_torch = torch.from_numpy(R.copy()).float().to(device)
        t_torch = torch.from_numpy(t.copy()).float().to(device)
        s_torch = torch.tensor(s).float().to(device)

        if torch.isnan(R_torch).any() or torch.isinf(R_torch).any():
            #set rotation error 180
            R_torch = torch.tensor([[ -1.,  0.,  0.],
                            [ 0., -1.,  0.],
                            [ 0.,  0., 1.]],
                           device=R_torch.device,
                           dtype=R_torch.dtype)
        if torch.isnan(t_torch).any() or torch.isinf(t_torch).any():
            #set translation error [5000,5000,5000]
            t_torch = torch.tensor([5000, 5000, 5000],
                           device=device,
                           dtype=t_torch.dtype)
        if torch.isnan(s_torch).any() or torch.isinf(s_torch).any():
            #set scale 40000
            s_torch = torch.tensor(40000.0,
                           device=s_torch.device,
                           dtype=s_torch.dtype)

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
        
        self.reg_loss_val = reg_loss
        self.norm_loss_val = norm_loss
        
        del(solver)
        combined_loss = self.beta * reg_loss + self.gamma * norm_loss
        return combined_loss

    def get_loss(self):
        r"""
        returns reconstruction loss, regression loss, norm loss
        """
        return self.rec_loss_val, self.reg_loss_val, self.norm_loss_val

    def set_weights(self, alpha,beta,gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma