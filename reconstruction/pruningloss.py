import torch
import torch.nn as nn
import numpy as np

from geomloss import SamplesLoss

import networkx as nx

import open3d as o3d

from math import sqrt
from scipy.linalg import sqrtm
from sklearn.cluster import DBSCAN

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

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_centroid(points):
    return np.mean(points,axis=0)

def calc_covariance(points,centroid):
    diffs = points - centroid                        
    covariance_matrix = np.dot(diffs.T, diffs) / len(points)
    return covariance_matrix


def make_psd(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0, a_max=None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def wasserstein_distance(mean1, cov1, mean2, cov2):
    mean_diff = np.linalg.norm(mean1 - mean2)**2
    
    cov1 = make_psd(cov1)
    cov2 = make_psd(cov2)
    
    sqrt_cov1 = sqrtm(cov1 + 1e-6 * np.eye(cov1.shape[0]))  # Regularization for stability
    cov_prod = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    
    trace_term = np.trace(cov1 + cov2 - 2 * cov_prod)
    
    return np.sqrt(mean_diff + trace_term)

class ReconstructLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wd = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        # self.uni

    def forward(self, p_q, p_gt):
        L = self.wd(p_q,p_gt[:,:3])
        # L += self.uni(p_q)
        
        return L

class PruingLoss(nn.Module):
    def __init__(
                 self, 
                 eps = 1, 
                 corr_threshold = 150,
                 ratio = 0.3,
                #  radius = 30
                 ):
        super().__init__()
        self.L_rec = ReconstructLoss()
        # self.r = radius
        self.ratio = ratio
        self.corr_threshold = corr_threshold
        self.eps = eps
        self.wd = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)

    
    def set_map(self, map, distance_threshold=0.5):
        r"""
        call when the sequence changed, call with func clear_map() for memory saving

        set semantic map
        - generate KD tree map for calculating sphere space points
        - generate map clusters
        - generate map triangles
        - generate map dictionary
        """
        assert isinstance(map, np.ndarray), "ASSERT: map type should be ndarray"

        self.map = map  # [n,4] n: number of pointclouds, 4:[x,y,z,label]

        clusters = self._closest_clustering(self.map)
        clusters = self._calc_cluster(clusters)
        # print(type(clusters))
        self.map_triangles = self._trianglize_query(clusters)
        # print("map triangle saved!")

    def clear_map(self):
        r"""
        remove unusable map data
        """
        del(self.map)
        del(self.pcd_map)
        del(self.kd_map)

    def _closest_clustering(self, points, distance_threshold=1):
        """
        라벨별로 분리한 뒤 DBSCAN으로 (x,y,z) 클러스터링.
        (points: (N,4) = [x,y,z,label])
        """
        labels = points[:, -1]
        unique_labels = np.unique(labels)

        clusters = []
        for label in unique_labels:
            label_mask = (labels == label)
            # (M,4) shape
            points_label = points[label_mask, :]  
            if points_label.shape[0] < 2:
                continue

            # DBSCAN
            coords_3d = points_label[:, :3]  # (M,3)
            clustering = DBSCAN(eps=distance_threshold, min_samples=20).fit(coords_3d)
            cluster_ids = clustering.labels_

            unique_cluster_ids = np.unique(cluster_ids)
            for cid in unique_cluster_ids:
                if cid == -1:  # noise
                    continue
                c_mask = (cluster_ids == cid)
                # (K,4)
                cluster_4d = points_label[c_mask, :]
                clusters.append(cluster_4d)

        print("[_closest_clustering] Number of clusters:", len(clusters))
        return clusters


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
        # print(query_gt[0:10])
        # Extract labels and identify cluster boundaries
        clusters = self._closest_clustering(query_gt)
        clusters = self._calc_cluster(clusters)  # Process clusters to add centroids and covariance
        query_triangles = self._trianglize_query(clusters)

        print(" Making I_raw ... ")
        i = 0
        for key in query_triangles.keys():
            if key in self.map_triangles.keys():
                i += 1

        print("whole number of quried points", i)
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
                        # Create correspondence with 3 pairs (0, 1, 2)
                        correspondence = [
                            (
                                (q_triangle["points"][i], q_triangle["centroids"][i], q_triangle["covariances"][i]),
                                (m_triangle["points"][i], m_triangle["centroids"][i], m_triangle["covariances"][i])
                            )
                            for i in range(3)
                        ]
                        I_raw.extend(correspondence)
        return I_raw
    
    def _similar(self, q_triangle, m_triangle):
        r"""
        input:
            - q_triangle: query triangle
            triangle = {
                    "points" : [points[key], points[key1], points[key2]],
                    "centroids": [centroids[key], centroids[key1], centroids[key2]],
                    "covariances": [covariances[key], covariances[key1], covariances[key2]],
                    "labels": [labels[key], labels[key1], labels[key2]],
                    "side_lengths": sorted_distances,
                }
            - m_triangle: map triangle
            triangle = {
                    "points" : [points[key], points[key1], points[key2]],
                    "centroids": [centroids[key], centroids[key1], centroids[key2]],
                    "covariances": [covariances[key], covariances[key1], covariances[key2]],
                    "labels": [labels[key], labels[key1], labels[key2]],
                    "side_lengths": sorted_distances,
                }
        output: boolean - return True if query and map are similar
        """
        avg = 0
        for i in range(3):
            # print("query triangle: ",q_triangle)
            # print("map triangle: ",m_triangle)
            if q_triangle["labels"][i] != m_triangle["labels"][i]:    # if the label isn't same
                return False
            
            avg += wasserstein_distance(q_triangle["centroids"][i],q_triangle["covariances"][i],m_triangle["centroids"][i],m_triangle["covariances"][i])

        return avg/3 < self.corr_threshold
    
    def _trianglize_query(self, clusters_dict, k=2, round_p = 0):
        """
        clusters_dict: { idx: [points_3d, label, centroid, covariance], ... }
        """
        from collections import defaultdict
        triangles = defaultdict(list)

        # 추출
        centroids = {cid: val[2] for cid, val in clusters_dict.items()}
        labels = {cid: val[1] for cid, val in clusters_dict.items()}
        covs = {cid: val[3] for cid, val in clusters_dict.items()}
        points_data = {cid: val[0] for cid, val in clusters_dict.items()}

        # 각 cluster의 centroid와 k개의 가까운 이웃으로 삼각형 만들기
        cluster_ids = list(centroids.keys())
        for key in cluster_ids:
            c_q = centroids[key]
            distances = []
            for other_key in cluster_ids:
                if other_key == key:
                    continue
                d = dist(c_q, centroids[other_key])
                distances.append((d, other_key))

            distances.sort(key=lambda x: x[0])
            closest_k = distances[:k]

            # 2개씩 조합
            from itertools import combinations
            for (d1, key1), (d2, key2) in combinations(closest_k, 2):
                d3 = dist(centroids[key1], centroids[key2])

                # float round (ex: 소수점 2자리)
                d1_r = round(d1, round_p)
                d2_r = round(d2, round_p)
                d3_r = round(d3, round_p)

                sorted_dist = tuple(sorted([d1_r, d2_r, d3_r]))

                triangle = {
                    "points": [
                        points_data[key], points_data[key1], points_data[key2]
                    ],
                    "centroids": [
                        centroids[key], centroids[key1], centroids[key2]
                    ],
                    "covariances": [
                        covs[key], covs[key1], covs[key2]
                    ],
                    "labels": [
                        labels[key], labels[key1], labels[key2]
                    ],
                    "side_lengths": sorted_dist
                }

                triangles[sorted_dist].append(triangle)
        return triangles

    def _calc_cluster(self, clusters):
        r"""
        input: semantic clustered pointcloud (list [n,4] n: points, 4:[x,y,z,label])
        output: dictionary {cluster idx : points, label, centroid, covariance matrix}
        """
        cluster_data = {}
        # print(clusters)
        for i,cluster in enumerate(clusters):
            if not cluster.size:
                continue

            centroid = calc_centroid(cluster[:, :3])
            covariance = calc_covariance(cluster[:, :3], centroid)
            cluster_data[i] = [cluster[:, :3], cluster[0, -1], centroid, covariance]

        # print(cluster_data.keys())
        return cluster_data
            
    def _get_I_pruned_map(self,I_raw):
        r"""
        Calc maximum clique of I_raw correspondence
        G = [V,E] V: correspondence of map cluster and query cluster edge difference function(vary) results 
        input: cluster tuple list [(query cluster, map cluster) ... ]
        output: I_pruned_map - list of map clusters that are part of the maximum clique

        triangle = {
            "points" : [points[key], points[key1], points[key2]],
            "centroids": [centroids[key], centroids[key1], centroids[key2]],
            "covariances": [covariances[key], covariances[key1], covariances[key2]],
            "labels": [labels[key], labels[key1], labels[key2]],
            "side_lengths": sorted_distances,
        }
        """
        
        # Triangle corr to cluster corr
        G = nx.Graph()
        for i, correspondence in enumerate(I_raw):
            G.add_node(i, data=correspondence)
        
        for i in range(len(I_raw)):
            for j in range(i + 1, len(I_raw)):
                if self._consistency_check(I_raw[i], I_raw[j]):
                    G.add_edge(i, j)
        cliques = list(nx.find_cliques(G))  
        max_clique = max(cliques, key=len)

        I_pruned_map = [G.nodes[node]["data"][0] for node in max_clique]

        return I_pruned_map

    def _consistency_check(self, corr1, corr2):
        r"""
        consistency check function
        input(tuple) : correlation cluster corr := (cluster_query, cluster_map)
            - cluster_query = (point, centroid, covariance)
        """
        query_diff = dist(corr1[0][1],corr2[0][1])
        map_diff = dist(corr1[1][1],corr2[1][1])

        diff = abs(query_diff - map_diff)
        return diff < self.eps

    # def _triangle_loss(self, query, query_gt):

    #     assert isinstance(query_gt, np.ndarray), "ASSERT: input type shoud be ndarray"
    #     # assert isinstance(query, list), "ASSERT: input type shoud be list"

    #     # P_q = self.model(query) # semantic segmented query pointcloud
    #     # self.P_q = self._calc_cluster(P_q)

    #     I_raw = self._get_I_raw(query_gt)
    #     I_pruned_map = self._get_I_pruned_map(I_raw)
    #     # del(P_q,I_raw)
    #     print(I_pruned_map.shape)

    #     map_points = []
    #     for cluster in I_pruned_map:
    #         map_points.extend(cluster.tolist())

    #     query = torch.tensor(query)
    #     map_points = torch.tensor(map_points)
        
    #     loss = self.L_rec(query, map_points)

    #     return loss
    
    def get_processed_map(self, query):
        I_raw = self._get_I_raw(query)
        I_pruned_map = self._get_I_pruned_map(I_raw)

        # print(I_pruned_map.shape)
        # print(I_pruned_map
        map_points = []
        # print(len(I_pruned_map))
        for cluster in I_pruned_map:
            print(cluster[0].shape)
            map_points.extend(cluster[0].tolist())

        return map_points


    # def _calc_triangle_loss(self, P_q :dict , I_pruned_map: list[dict]):
    #     r"""
    #     input: 
    #         - I_pruned_map (list) : map clusters [map_cluster1, map cluster2 ... ]  
    #         - P_q (dict) : semantic segmented queried pointcloud dictionary
    #         cluster = {index: [points, label, centroid, covariance]}
    #     output: similariy loss - similar cluster's reconstruction loss, distance of each cluster 
    #     """
    #     correspond = self._map_similar_clusters() #Calc by softmax
    #     return loss

    # def _map_similar_clusters(self):
    #     r"""
    #     mapping query cluster and map cluster by nearest centroid
    #     """
    #     num_query_clusters = len(self.P_q)
    #     num_map_clusters = len(self.I_pruned_map)

    #     correspond = []
    #     for k_q, c_q in self.P_q.items():
    #         distance = []
    #         for i,c_m in enumerate(self.I_pruned_map):
    #             distance.append((dist(c_q[2],c_m[2]),i))

    #         distance.sort()
    #         correspond.append((k_q , distance[0][1]))  # (query index, map_index)

    #     return correspond


    def forward(self, P_r, P_gt):
        r"""
        P_r : reconstructed pointcloud (not semantic segmented) [M,3]
        P_gt: gt semantic segmented pointcloud 64 channel [N,4]
        """
        # L_upsample = self.L_rec(P_r, P_gt)    # upsampling loss
        L_tri = self._triangle_loss(P_r.numpy(), P_gt.numpy())
        return L_tri