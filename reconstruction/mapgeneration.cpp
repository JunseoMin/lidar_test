#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ------------------------- (A) Data Structures -------------------------

// (x, y, z, label)
using PointT = pcl::PointXYZL;
using CloudT = pcl::PointCloud<PointT>;

// A small struct that keeps track of a PCL point & its original index
struct IndexedPoint {
    PointT pt;
    size_t original_index; 
};

// Tolerance
static const float CLUSTER_TOLERANCE = 2.0f;  
static const int   MIN_CLUSTER_SIZE  = 5;
static const int   MAX_CLUSTER_SIZE  = 1000000;

// ------------------------- (B) Save centroids in bin -------------------------
void saveCentroidsBin(const std::vector<Eigen::Vector3f> &centroids, 
                      const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if(!ofs.is_open()){
        std::cerr << "[Error] Cannot open file for writing: " << filename << std::endl;
        return;
    }

    for(const auto &c : centroids){
        ofs.write(reinterpret_cast<const char*>(c.data()), sizeof(float)*3);
    }
    ofs.close();
    std::cout << "[Info] Saved " << centroids.size() 
              << " centroids to " << filename << std::endl;
}

// ------------------- (C) Save covariance + normal + indices in txt -------------------
/**
 * @param covariances       N 3x3 covariance matrices
 * @param normals           N normals (3D)
 * @param subcloudIndices   N sets of pcl::PointIndices (indices w.r.t. each sub-cloud)
 * @param originalIndices   N sets of original indices (w.r.t. the full input cloud)
 * @param clusterLabels     N cluster labels
 * @param clusterLocalIDs   N cluster IDs local to each label
 * @param filename          output txt file
 */
void saveCovarianceTxt(const std::vector<Eigen::Matrix3f> &covariances,
                       const std::vector<Eigen::Vector3f> &normals,
                       const std::vector<pcl::PointIndices> &subcloudIndices,
                       const std::vector<std::vector<size_t>> &originalIndices,
                       const std::vector<uint32_t> &clusterLabels,
                       const std::vector<int> &clusterLocalIDs,
                       const std::string &filename)
{
    if (covariances.size() != normals.size() ||
        covariances.size() != subcloudIndices.size() ||
        covariances.size() != originalIndices.size() ||
        covariances.size() != clusterLabels.size() ||
        covariances.size() != clusterLocalIDs.size())
    {
        std::cerr << "[Warning] size mismatch in saveCovarianceTxt!\n";
    }

    std::ofstream ofs(filename);
    if(!ofs.is_open()){
        std::cerr << "[Error] Cannot open file for writing: " << filename << std::endl;
        return;
    }

    for(size_t i = 0; i < covariances.size(); i++){
        ofs << "ClusterIdx: " << i 
            << " Label: " << clusterLabels[i]
            << " LocalClusterID: " << clusterLocalIDs[i] << "\n";
        
        // Covariance
        // ofs << "Cov:\n" << covariances[i] << "\n";

        // Indices in the subcloud
        // ofs << "Subcloud Indices (size=" << subcloudIndices[i].indices.size() << "): ";
        // for (auto idx : subcloudIndices[i].indices) {
        //     ofs << idx << " ";
        // }
        // ofs << "\n";

        // Original Indices in the full cloud
        ofs << "Original Indices: \n";
        for (auto orig : originalIndices[i]) {
            ofs << orig << "\n";
        }
        ofs << "\n";

        // Normal
        // ofs << "Normal: " 
        //     << normals[i](0) << " "
        //     << normals[i](1) << " "
        //     << normals[i](2) << "\n\n";    
    }
    ofs.close();
    std::cout << "[Info] Saved " << covariances.size()
              << " covariance+normal+indices to " << filename << std::endl;
}

// ------------------------- (D) Load BIN file -------------------------
bool loadBinFile(const std::string &input_file, CloudT::Ptr &cloud_out)
{
    std::ifstream bin_file(input_file, std::ios::binary);
    if(!bin_file.is_open()){
        std::cerr << "[Error] Could not open input file: " << input_file << std::endl;
        return false;
    }

    while(true){
        float x=0.f, y=0.f, z=0.f, label_f=0.f;
        // read 4 floats
        bin_file.read(reinterpret_cast<char*>(&x), sizeof(float));
        if(!bin_file.good()) break;

        bin_file.read(reinterpret_cast<char*>(&y), sizeof(float));
        if(!bin_file.good()) break;

        bin_file.read(reinterpret_cast<char*>(&z), sizeof(float));
        if(!bin_file.good()) break;

        bin_file.read(reinterpret_cast<char*>(&label_f), sizeof(float));
        if(!bin_file.good()) break;

        uint32_t label_u = static_cast<uint32_t>(label_f);

        PointT pt;
        pt.x = x;
        pt.y = y;
        pt.z = z;
        pt.label = label_u; 
        cloud_out->points.push_back(pt);
    }

    bin_file.close();
    cloud_out->width = cloud_out->points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;

    std::cout << "[Info] Loaded " << cloud_out->points.size() 
              << " points from " << input_file << std::endl;
    return true;
}

// ------------------------- (E) Main logic: label-based clustering + centroid + cov + normal -------------------------
void processPointCloud(const std::string &input_file, 
                       const std::string &centroids_bin,
                       const std::string &cov_txt)
{
    // 1) Load the original cloud
    CloudT::Ptr cloud(new CloudT);
    if(!loadBinFile(input_file, cloud)){
        return;
    }
    
    // 2) Create a map from label -> vector<IndexedPoint>
    //    We'll keep track of (PointT + original index) for each label.
    std::unordered_map<uint32_t, std::vector<IndexedPoint>> labelMap;
    labelMap.reserve(50);

    // Fill labelMap
    for(size_t i = 0; i < cloud->points.size(); i++){
        const auto &pt = cloud->points[i];
        labelMap[pt.label].push_back({pt, i}); 
    }

    // These vectors will hold final results from all clusters of all labels
    std::vector<Eigen::Vector3f> all_centroids;
    std::vector<Eigen::Matrix3f> all_covs;
    std::vector<Eigen::Vector3f> all_normals;
    std::vector<pcl::PointIndices> all_subcloud_indices;  // subcloud-based indices
    std::vector<std::vector<size_t>> all_original_indices; // original cloud indices

    std::vector<uint32_t> clusterLabels; 
    std::vector<int>      clusterLocalIDs;    

    // Unique labels
    std::vector<uint32_t> uniqueLabels;
    uniqueLabels.reserve(labelMap.size());
    for(auto &kv : labelMap){
        uniqueLabels.push_back(kv.first);
    }

#ifdef _OPENMP
    std::cout << "[Info] Using OpenMP with " << omp_get_max_threads() << " threads.\n";
#endif

    // 3) For each label, run Euclidean clustering, compute centroid/cov/normal
#pragma omp parallel
    {
        // Thread-local accumulators
        std::vector<Eigen::Vector3f> local_centroids;
        std::vector<Eigen::Matrix3f> local_covs;
        std::vector<Eigen::Vector3f> local_normals;
        std::vector<pcl::PointIndices> local_subcloud_indices;
        std::vector<std::vector<size_t>> local_original_indices;

        std::vector<uint32_t> local_labels;
        std::vector<int>      local_clusterIDs;

#pragma omp for schedule(dynamic)
        for(size_t i=0; i<uniqueLabels.size(); i++)
        {
            uint32_t labelVal = uniqueLabels[i];
            // std::cout << labelVal <<std::endl;
            if (labelVal != 50 && labelVal != 80 && labelVal != 81 && labelVal != 71) {
                continue; // Skip the rest of the loop iteration
            }
            // std::cout << labelVal;
            // Build a subcloud for clustering
            CloudT::Ptr labelCloud(new CloudT);
            labelCloud->points.reserve(labelMap[labelVal].size());

            // Just to ensure PCL clustering works, copy points in
            for(const auto &ipt : labelMap[labelVal]) {
                labelCloud->points.push_back(ipt.pt);
            }

            if(labelCloud->points.empty()) continue;

            // KdTree + Euclidean Clustering
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            tree->setInputCloud(labelCloud);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(CLUSTER_TOLERANCE);
            ec.setMinClusterSize(MIN_CLUSTER_SIZE);
            ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
            ec.setSearchMethod(tree);
            ec.setInputCloud(labelCloud);
            ec.extract(cluster_indices);

            // For each found cluster
            int localID = 0;
            for(const auto &inds : cluster_indices)
            {
                // 3.a) Build a small cluster
                CloudT::Ptr cluster(new CloudT);
                cluster->points.reserve(inds.indices.size());

                // We'll also map subcloud indices -> original indices
                std::vector<size_t> cluster_orig_indices;
                cluster_orig_indices.reserve(inds.indices.size());

                for(int idx : inds.indices) {
                    // subcloud index -> original index
                    size_t orig_idx = labelMap[labelVal][idx].original_index;

                    // push the actual point
                    cluster->points.push_back(labelCloud->points[idx]);
                    // keep track of the original index
                    cluster_orig_indices.push_back(orig_idx);
                }

                // 3.b) centroid
                Eigen::Vector4f c4f;
                pcl::compute3DCentroid(*cluster, c4f);
                Eigen::Vector3f c3f = c4f.head<3>();

                // 3.c) covariance
                Eigen::Matrix3f cov;
                pcl::computeCovarianceMatrixNormalized(*cluster, c4f, cov);

                // 3.d) normal (smallest eigenvalue => normal direction)
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
                Eigen::Vector3f eigVals = solver.eigenvalues();
                Eigen::Matrix3f eigVecs = solver.eigenvectors();

                int minIdx = 0;
                float minVal = eigVals(0);
                for(int k=1; k<3; k++){
                    if(eigVals(k) < minVal){
                        minVal = eigVals(k);
                        minIdx = k;
                    }
                }
                Eigen::Vector3f normal = eigVecs.col(minIdx).normalized();

                // 3.e) Save cluster data to local accumulators
                local_centroids.push_back(c3f);
                local_covs.push_back(cov);
                local_normals.push_back(normal);
                local_subcloud_indices.push_back(inds); // subcloud-based
                local_original_indices.push_back(cluster_orig_indices); // original

                local_labels.push_back(labelVal);
                local_clusterIDs.push_back(localID++);
            }
        } // end for labels

        // Merge thread-local results into global vectors
#pragma omp critical
        {
            all_centroids.insert(all_centroids.end(),
                                 local_centroids.begin(), local_centroids.end());
            all_covs.insert(all_covs.end(),
                            local_covs.begin(), local_covs.end());
            all_normals.insert(all_normals.end(),
                               local_normals.begin(), local_normals.end());
            all_subcloud_indices.insert(all_subcloud_indices.end(),
                                        local_subcloud_indices.begin(), local_subcloud_indices.end());
            all_original_indices.insert(all_original_indices.end(),
                                        local_original_indices.begin(), local_original_indices.end());
            clusterLabels.insert(clusterLabels.end(),
                                 local_labels.begin(), local_labels.end());
            clusterLocalIDs.insert(clusterLocalIDs.end(),
                                   local_clusterIDs.begin(), local_clusterIDs.end());
        }
    } // end omp parallel

    // 4) Save results
    saveCentroidsBin(all_centroids, centroids_bin);

    // Save covariance, normal, and also subcloud/original indices
    saveCovarianceTxt(all_covs, 
                      all_normals,
                      all_subcloud_indices,
                      all_original_indices,
                      clusterLabels,
                      clusterLocalIDs,
                      cov_txt);

    std::cout << "[Info] Done. Found " << all_centroids.size()
              << " clusters in total.\n";
}

// --------------------(F) main------------------------
void printUsage(const std::string& program_name) {
    std::cout << "usage: " << program_name 
              << " <input_file> <output_centroids> <output_covariance>" 
              << std::endl;
}

int main(int argc, char* argv[]) 
{
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_file       = argv[1];
    std::string output_centroids = argv[2];
    std::string output_cov       = argv[3];

    try {
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/00_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/00/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/00/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/01_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/01/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/01/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/02_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/02/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/02/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/03_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/03/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/03/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/04_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/04/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/04/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/05_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/05/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/05/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/06_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/06/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/06/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/07_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/07/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/07/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/08_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/08/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/08/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/09_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/09/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/09/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/10_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/10/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/10/cov.txt");
        processPointCloud("/home/server01/js_ws/dataset/odometry_dataset/semantic_map/11_map.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/11/centroids.bin","/home/server01/js_ws/dataset/odometry_dataset/inlier_map/11/cov.txt");
        // processPointCloud(input_file,output_centroids,output_centroids);
        // processPointCloud(input_file, output_centroids, output_cov);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR]: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "process finished!" << std::endl;
    return 0;
}
