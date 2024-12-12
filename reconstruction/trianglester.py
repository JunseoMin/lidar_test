import numpy as np
from addict import Dict

class Triangle(Dict):
    r"""
    1. calc centroids
    2. calc distance
    3. map sigma&centroids
    """

    def __init__(self, c1,c2,c3):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
    

    def calc_distance(self):
        pass

    def get_triangle(self):
        pass

def calc_centroid():
    pass

def calc_variance():
    pass

def get_label():
    pass

class TriangleGenerator():

    def __init__(self, clustering_method):
        self.clustering_method = clustering_method
        
        pass

    def make_segmented(self,lidar_raw):
        segments = []

        return segments

    def triangulize(self,clusters):
        r"""
        generate triangle Graph
        """
        
        triangles = []

        for centroid, label, sigma in clusters:
            
            
            pass

        return triangles

    def generate_clusterlist(self,lidar_raw, is_map = False):
        r"""
        1. smentic segmentation
        2. calc things
        3. list them
        """

        clusters = []   # contain tuple(centroid, label, sigma) list
        segments = self.make_segmented(lidar_raw)
        
        for seg in segments:
            centroid = calc_centroid(seg)
            label = get_label(seg)
            sigma = calc_variance(seg)

            clusters.append((centroid,label,sigma))

        return clusters

    