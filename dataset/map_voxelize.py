import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/junseo/MPIL/implementations/lidar_sr/map/gt_map.pcd")

# Optional: Downsample the map for efficiency
voxelized = pcd.voxel_down_sample(voxel_size=0.1)
print("voxelized!")


o3d.io.write_point_cloud("/home/junseo/MPIL/implementations/lidar_sr/map/voxelized.pcd", voxelized)