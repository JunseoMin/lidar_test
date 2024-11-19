import numpy as np
import torch
import torch.nn as nn

from einops import rearrange

class bbox(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox
    
    def forward(lidar_raw):
        r"""
        lidar_raw : raw data(torch.tensor() type) n*4
        make bounding box match with the scan range

        return : bbox((min,max) X3)
        """
        
        lidar_x = lidar_raw[:][0]
        lidar_y = lidar_raw[:][1]
        lidar_z = lidar_raw[:][2]

        mm_lidar_x = tuple(torch.aminmax(lidar_x))
        mm_lidar_y = tuple(torch.aminmax(lidar_y))
        mm_lidar_z = tuple(torch.aminmax(lidar_z))

        bbox = (mm_lidar_x, mm_lidar_y, mm_lidar_z)

        return bbox