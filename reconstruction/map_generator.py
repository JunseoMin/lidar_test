import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

# File paths
input_bin_file = "/home/server01/js_ws/dataset/odometry_dataset/gt_map/04_map.bin"  # Input [N, 4] file
output_centroids_file = "/home/server01/js_ws/dataset/odometry_dataset/featmap/centroids.pcd"  # Output [M, 3] file
output_covariance_file = "/home/server01/js_ws/dataset/odometry_dataset/featmap/covariance.txt"  # Output covariance for each cluster

# Load the binary file
# Each row is [x, y, z, label]
data = np.fromfile(input_bin_file, dtype=np.float32).reshape(-1, 4)
points = data[:, :3]  # Extract x, y, z
labels = data[:, 3].astype(int)  # Extract labels

# Process each label
unique_labels = np.unique(labels)
centroids = []
covariances = []

cluster_id = 0
for label in unique_labels:
    # Get points with the current label
    label_mask = (labels == label)
    label_points = points[label_mask]
    
    # Cluster these points spatially (DBSCAN for spatial clustering)
    clustering = DBSCAN(eps=2.0, min_samples=5).fit(label_points)
    cluster_labels = clustering.labels_
    
    # Process each spatial cluster
    for cluster in np.unique(cluster_labels):
        if cluster == -1:  # Ignore noise points
            continue
        
        cluster_mask = (cluster_labels == cluster)
        cluster_points = label_points[cluster_mask]
        
        # Compute centroid
        centroid = np.mean(cluster_points, axis=0)
        
        # Compute covariance matrix
        covariance = np.cov(cluster_points.T)
        
        # Save results
        centroids.append(centroid)
        covariances.append((cluster_id, label, centroid, covariance))
        cluster_id += 1

# Save centroids as [M, 3]
centroids = np.array(centroids)
centroid_pcd = o3d.geometry.PointCloud()
centroid_pcd.points = o3d.utility.Vector3dVector(centroids)
o3d.io.write_point_cloud(output_centroids_file, centroid_pcd)


# Save covariance matrices for each cluster
with open(output_covariance_file, "w") as f:
    for cluster_id, label, centroid, covariance in covariances:
        f.write(f"Cluster ID: {cluster_id}, Label: {label}\n")
        f.write(f"Centroid: {centroid.tolist()}\n")
        f.write(f"Covariance:\n{covariance}\n\n")

print(f"Centroids saved to {output_centroids_file}")
print(f"Covariance matrices saved to {output_covariance_file}")
