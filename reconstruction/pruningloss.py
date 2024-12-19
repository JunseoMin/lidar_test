import torch
import torch.nn as nn
import numpy as np

from trianglester import Triangle
from geomloss import SamplesLoss

import networkx as nx

import open3d as o3d

from math import sqrt
from scipy.linalg import sqrtm


r"""
Notations:
    P_r : Lidar reconstructed
    P_gt : GT LiDAR (64ch)
    pose_gt: GT pose

TODO:
    1. class Triangle(?) -- DONE (12/19)
    2. I_raw calculation -- DONE (12/19)
    3. maximum clique function
    4. reconstruction loss (query, map)
    BIG5. !DEBUG!
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


def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    r"""
    Calculate the 2-Wasserstein distance between two Gaussian distributions.
    
    Parameters:
        mu1: Mean vector of the first distribution (1D numpy array)
        sigma1: Covariance matrix of the first distribution (2D numpy array)
        mu2: Mean vector of the second distribution (1D numpy array)
        sigma2: Covariance matrix of the second distribution (2D numpy array)
    
    Returns:
        Wasserstein distance (float)
    """
    # Compute the Euclidean distance between the means
    mean_diff = np.linalg.norm(mu1 - mu2)**2
    
    # Compute the square root of the first covariance matrix
    sigma1_sqrt = sqrtm(sigma1)
    
    # Ensure the result is real-valued (handle numerical instability)
    if np.iscomplexobj(sigma1_sqrt):
        sigma1_sqrt = sigma1_sqrt.real
    
    # Compute the matrix product for the covariance term
    covariance_term = sqrtm(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
    
    # Ensure the result is real-valued
    if np.iscomplexobj(covariance_term):
        covariance_term = covariance_term.real
    
    # Trace terms
    trace_term = np.trace(sigma1 + sigma2 - 2 * covariance_term)
    
    # Wasserstein distance
    wasserstein_dist = mean_diff + trace_term
    return np.sqrt(wasserstein_dist)

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
    def __init__(self, corr_threshold = 0.5, ratio = 0.3 ,radius = 30):
        super().__init__()
        self.L_rec = ReconstructLoss()
        self.r = radius
        self.ratio = ratio
        self.corr_threshold = corr_threshold
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
        Generate raw correspondence list I_raw based on query and map triangles.
        """

        # Extract labels and identify cluster boundaries
        labels = query_gt[:, -1]
        label_changes = np.where(labels[:-1] != labels[1:])[0] + 1

        # Split query points into clusters and process them
        clusters = np.split(query_gt, label_changes)
        clusters = self._calc_cluster(clusters)  # Process clusters to add centroids and covariance
        query_triangles = self._trianglize_query(clusters)

        # Release memory for clusters
        del clusters

        # Initialize I_raw
        I_raw = list()

        # Match query triangles with map triangles and calculate correspondences
        for key, q_triangles in query_triangles.items():
            map_triangles = self.map_triangles.get(key, [])
            for q_triangle in q_triangles:
                for m_triangle in map_triangles:
                    if self._similar(q_triangle, m_triangle):
                        # Create correspondence tuple for each vertex
                        correspondence = [(q_triangle[i], m_triangle[i]) for i in range(3)]
                        I_raw.extend(correspondence)

        return I_raw
    
    def _similar(self, q_triangle, m_triangle):
        r"""
        input:
            - q_triangle: query triangle (list [cluster1, cluster2, cluster3])
            - m_triangle: map triangle (list [cluster1, cluster2, cluster3])
            i.e.) cluster = [points, label, centroid, covariance matrix]
        output: boolean - return True if query and map are similar
        """

        for i in range(3):
            if q_triangle[i][1] != m_triangle[i][1]:    # if the label isn't same
                return False
            wd = wasserstein_distance(q_triangle[2],q_triangle[-1],m_triangle[2],m_triangle[-1])
            if wd > self.corr_threshold:
                return False

        return True
    
    def _trianglize_query(self, clusters):
        r"""
        Make triangles by cluster's centroids with 2 nearest neighbor clusters.
        Input: clusters (dictionary)
        Output: triangles (dictionary: {tuple of sorted distances: [c1, c2, c3]})
        """

        triangles = {}
        centroids = {key: val[2] for key, val in clusters.items()}  # Extract centroids

        for key, c_q in centroids.items():
            # Calculate distances to other centroids
            distances = [
                (dist(c_q, c_other), other_key)
                for other_key, c_other in centroids.items()
                if key != other_key
            ]

            # Sort by distance and get the two closest clusters
            distances.sort()
            closest_two = [distances[0], distances[1]]  # [(distance, index), ...]

            # Extract distances and sort them to form a unique key
            d1 = closest_two[0]  # Distance to the first nearest cluster
            d2 = closest_two[1]  # Distance to the second nearest cluster
            d3 = dist(
                centroids[closest_two[0][1]], centroids[closest_two[1][1]]
            )  # Distance between the two nearest clusters

            triangle_key = tuple(sorted([d1, d2, d3]))

            # Add triangle to the dictionary
            if triangle_key not in triangles:
                triangles[triangle_key] = [
                    clusters[triangle_key[0][1]],   # corresponds to sorted index
                    clusters[triangle_key[1][1]],   
                    clusters[triangle_key[2][1]]    
                ]

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
        r"""
        Calc maximum clique of I_raw correspondence
        G = [V,E] V: correspondence of map cluster and query cluster edge difference function(vary) results 
        input: cluster tuple list [(query cluster, map cluster) ... ]
        output: I_pruned - list of correspondences that are part of the maximum clique
        """

        G = nx.Graph()
        for i, correspondence in enumerate(I_raw):
            G.add_node(i, data=correspondence)
        
        for i in range(len(I_raw)):
            for j in range(i + 1, len(I_raw)):
                if self._consistency_check(I_raw[i], I_raw[j]):
                    G.add_edge(i, j)
        cliques = list(nx.find_cliques(G))  
        max_clique = max(cliques, key=len)

        I_pruned = [G.nodes[node]["data"] for node in max_clique]

        return I_pruned

    def _consistency_check(self, corr1, corr2):
        r"""
        TODO
        consistency check function
        """
        query_diff = abs(corr1[0] - corr2[0])
        map_diff = abs(corr1[1] - corr2[1])
        return query_diff <= 1 and map_diff <= 1 

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
