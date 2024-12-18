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

TODO:
    1. class Triangle(?)
    2. I_raw calculation
    3. maximum clique function
    4. reconstruction loss (query, map)
"""

def dist(p1,p2):
    dist = 0
    for a,b in p1,p2:
        dist += (a - b)**2
    return sqrt(dist)

def calc_centroid(points):
    return np.mean(points,axis=0)

def calc_covariance(points,centroid):
    diffs = points - centroid                        
    covariance_matrix = np.dot(diffs.T, diffs) / len(points)
    return covariance_matrix


class ReconstructLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wd = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        self.uni

    def forward(self, p_q, p_gt):
        L = self.wd(p_q,p_gt[:,:3])
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
        call when the sequence changed, call with func clear_map() for memory saving

        set semantic map
        - generate KD tree map for calculating sphere space points
        - generate map clusters
        - generate map triangles
        - generate map dictionary
        """
        assert isinstance(map, np.ndarray), "ASSERT: map type shoud be ndarray"

        self.map = map  # [n,4] n: number of pointclouds, 4:[x,y,z,label]
        self.pcd_map = o3d.geometry.PointCloud()

        self.pcd_map.points = o3d.utility.Vector3dVector(map[:,:3])
        self.kd_map = o3d.geometry.KDTreeFlann(self.pcd_map)

        labels = map[:,:-1]
        label_changes = np.where(labels[:-1] != labels[1:])[0] + 1

        self.map_clusters = np.split(map, label_changes)
        self.map_clusters = self._calc_cluster(self.map_clusters)
        self.map_triangles = self._trianglize_query(self.map_clusters)

    def clear_map(self):
        r"""
        remove unusable map data
        """
        del(self.map)
        del(self.pcd_map)
        del(self.kd_map)
    
    def _get_gt_cluster_map_sphere(self, pose):
        r"""
        get points in the sphere space where the GT pose given
        """
        t = pose[:3, 3]   # Translation vector (top-right 3x1)
        indices = self.kd_map.search_radius_vector_3d(t,self.r)[1]   # indices index list
        gt_pcd = self.map[indices]
        return gt_pcd

    def _get_I_raw(self, query_gt):
        r"""
        generate raw correspondence list I_raw
        """

        I_raw = None
        
        labels = query_gt[:, -1]
        label_changes = np.where(labels[:-1] != labels[1:])[0] + 1 

        clusters = np.split(query_gt, label_changes)

        clusters = self._calc_cluster(clusters)
        query_triangles = self._trianglize_query(clusters)
        
        #calc I_raw



        return I_raw
    
    def _trianglize_query(self, clusters):
        r"""
        Make triangle by cluster's centroid with 2 nearest neighbor clusters
        input: cluster (dictionary)
        output: triangle (dictionary : {main cluster's idx : {indexes of two nearest clusters}})
        """

        triangles = {}
        centroids = {key: val[2] for key, val in clusters.items()}

        for key, c_q in centroids.items():
            distances = []
            for other_key, c_other in centroids.items():
                if key != other_key:
                    distance = dist(c_q, c_other)
                    distances.append((distance, other_key))
            
            distances.sort()
            closest_two = distances[:2]

            triangles[key] = [key] + [item[1] for item in closest_two]

        return triangles

    def _calc_cluster(self, clusters):
        r"""
        input: semantic clustered pointcloud (list [n,4] n: points, 4:[x,y,z,label])
        output: dictionary {cluster idx : points, label, centroid, covariance matrix}
        """
        cluster_data = {}

        for i,cluster in enumerate(clusters):
            centroid = calc_centroid(cluster[:, :3])
            covariance = calc_covariance(cluster[:, :3], centroid)
            cluster_data[i] = [cluster[:, :3], cluster[0, -1], centroid, covariance]

        return cluster_data
            
    def _get_I_pruned(self,I_raw):
        pass

    def triangle_loss(self, query, query_gt, pose):

        assert isinstance(query_gt, np.ndarray), "ASSERT: input type shoud be ndarray"
        assert isinstance(query, np.ndarray), "ASSERT: input type shoud be ndarray"
        assert isinstance(pose, np.ndarray), "ASSERT: input type shoud be ndarray"

        P_q = self.model(query) # semantic segmented query pointcloud

        I_raw = self._get_I_raw(query_gt)
        I_pruned = self._get_I_pruned(I_raw)

        loss = self._calc_triangle_loss(P_q, I_pruned)

        return loss

    def forward(self, P_r, P_gt, pose_gt):
        r"""
        P_gt: gt semantic segmented pointcloud
        pose_gt: GT pose
        P_r : reconstructed pointcloud (not semantic segmented)
        """

        L_upsample = self.L_rec(P_r, P_gt)    # upsampling loss
        L_tri = self.triangle_loss(P_r, P_gt, pose_gt)

        loss = L_upsample * self.ratio + L_tri * (1-self.ratio)
        return loss
