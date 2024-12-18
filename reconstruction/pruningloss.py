import torch
import torch.nn as nn
import numpy as np

from trianglester import TriangleGenerator
from geomloss import SamplesLoss

from networkx import Graph

import open3d as o3d

from math import sqrt

r"""
Notations:
    P_r : Lidar reconstructed
    P_gt : GT LiDAR (64ch)
    pose_gt: GT pose
"""

def dist(p1,p2):
    dist = 0
    for a,b in p1,p2:
        dist += (a - b)**2
    return sqrt(dist)


class ReconstructLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wd = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        self.uni

    def forward(self, p_q, p_gt):
        L = self.wd(p_q,p_gt)
        L += self.uni(p_q)
        
        return L


class PruingLoss(nn.Module):
    def __init__(self, ratio = 0.3 ,radius = 30):
        r"""
        r: radious of a sphere
        """
        super().__init__()
        self.L_rec = ReconstructLoss()
        self.r = radius
        self.ratio = ratio
        self.model = model()    # semantic segmentation model initialize
    
    def set_map(self, map):
        r"""
        set semantic map
        """
        self.map = map  # [n,4] n: number of pointclouds, 4:[x,y,z,label]
        self.pcd_map = o3d.geometry.PointCloud()

        self.pcd_map.points = o3d.utility.Vector3dVector(map[:,:3])
        self.kd_map = o3d.geometry.KDTreeFlann(self.pcd_map)

    def clear_map(self):
        del(self.map)
        del(self.pcd_map)
        del(self.kd_map)

    def calc_global_graph():
        pass
    
    def get_gt_cluster_map(self, pose):
        t = pose[:3, 3]   # Translation vector (top-right 3x1)
        indices = self.kd_map.search_radius_vector_3d(t,self.r)[1]   # indices index list
        
        gt_pcd = self.map[indices]
        return gt_pcd

    def reconstruct_loss(self, query, pose):
        P_q = self.model(query)
        P_gt = self.get_gt_cluster_map(pose)

        graph_q = graphify(P_q)
        graph_gt = graphify(P_gt)

        gt = pruning(graph_gt)  # calc maximum clique
        pass

    def forward(self, P_r, P_gt, pose_gt):
        r"""

        """
        L_upsample = self.L_rec(P_r,P_gt)    # upsampling loss
        L_tri = self.reconstruct_loss(P_r,pose_gt)

        loss = L_upsample * self.ratio + L_tri * (1-self.ratio)
        return loss
