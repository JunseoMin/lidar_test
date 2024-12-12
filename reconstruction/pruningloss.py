import torch
import torch.nn as nn
import numpy

from trianglester import TriangleGenerator

class PruingLoss(nn.Module):
    def __init__(self, map_hash):
        super().__init__()
        self.map_hash = map_hash
        self.generator = TriangleGenerator()

    def correlation_graphify(self, query_lidar):
        r"""
        1. Calcualate wasserstein distance between query triangle and Map triangel
        2. Generate consistancy graph which satisfying cosntraints
        return: I_raw
        """

        return

    def outlier_pruning(self, high_ch_raw):
        r"""
        Generate I_pruned through Maximum clique
        """
        
        pruned = []

        return pruned
    
    def calculate_loss(self, I_raw, I_pruned):
        r"""
        I_raw: the upsampling model output with correlated maps
        I_pruned: pruned data (GT)
        """

        return 0

    def forward(self, low_ch_upsampled_raw, high_ch_raw):
        r"""
        low_ch_upsampled: downsampled 16channel LiDAR input z_l
        high_ch_raw: raw 64channel LiDAR input z_h
        
        Object - generate upsampled LiDAR pointcloud which can generate pruned graph.
        loss - I = argmin error(I_raw, I_prun)
        """

        loss = 0.
        
        # make data the list [A1,A2, ... ]
        high_clusters = self.generator.generate_clusterlist(high_ch_raw)
        upsampled_clusters = self.generator.generate_clusterlist(low_ch_upsampled_raw)
        
        high_triangles = self.generator.triangulize(high_clusters)
        upsampled_triangles = self.generator.triangulize(upsampled_clusters)

        # Get I_raw = { (A1,Bi), (A2,Bj) ... }; B -> from map, A-> from Query
        I_raw_high = self.correlation_graphify(high_triangles)
        I_raw_upsampled = self.correlation_graphify(upsampled_triangles)

        I_pruned = self.outlier_pruning(I_raw_high) # Get GT for loss function
        
        loss = self.calculate_loss(I_raw_upsampled, I_pruned)   # loss with co-exist clusters
        
        return loss
