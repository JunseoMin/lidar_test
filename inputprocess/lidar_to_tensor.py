import torch
import torch.nn as nn

import numpy as np

from einops import rearrange
from torch.utils.data import Dataset
from glob import glob

class KiTTILoader(Dataset):
    def __init__(self,file_paths):
        r"""
        filepaths = path to your **.bin files
        """
        super().__init__()
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
            
    def __getitem__(self, idx):
        bin_path = self.file_paths[idx]
        points = self.load_vlp_points(bin_path)
        
        points_tensor = torch.from_numpy(points).float()
        
        lidar_x = points_tensor[:][0]
        lidar_y = points_tensor[:][1]
        lidar_z = points_tensor[:][2]

        mm_lidar_x = tuple(torch.aminmax(lidar_x))
        mm_lidar_y = tuple(torch.aminmax(lidar_y))
        mm_lidar_z = tuple(torch.aminmax(lidar_z))

        bbox = (mm_lidar_x, mm_lidar_y, mm_lidar_z)

        return points_tensor,bbox
    
    def load_vlp_points(self,bin_path):
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
        return points
