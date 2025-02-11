/***
 * @brief Generate centroids & normal .bin files from map.bin and cov.txt for GT dataset 
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>

// ------------------- Data structures -------------------
struct GlobalPoint {
    float x, y, z, label;
};

struct ClusterInfo {
    std::vector<size_t> original_indices;  // index into globalPoints
};

struct ClusterCenter {
    float x, y, z;      // centroid
    float nx, ny, nz;   // normal
    // float label;        // optional label
};

// ------------------- Utility: load poses (camera->map) -------------------
bool loadKITTIposes(const std::string &pose_file, std::vector<Eigen::Matrix4d> &poses)
{
    poses.clear();
    std::ifstream fin(pose_file);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open pose file: " << pose_file << std::endl;
        return false;
    }
    std::string line;
    while(std::getline(fin, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> vals;
        double v;
        while(ss >> v) vals.push_back(v);
        if(vals.size() != 12){
            std::cerr << "[Warning] a pose line doesn't have 12 floats\n";
            continue;
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for(int r=0; r<3; r++){
            for(int c=0; c<4; c++){
                T(r,c) = vals[r*4 + c];
            }
        }
        poses.push_back(T);
    }
    fin.close();
    return true;
}

// ------------------- Utility: load global_map.bin => [x,y,z,label] float -------------------
bool loadGlobalMap(const std::string &filename, std::vector<GlobalPoint> &globalPoints){
    globalPoints.clear();
    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open global map: " << filename << std::endl;
        return false;
    }
    while(true){
        float x=0,y=0,z=0, label=0;
        if(!fin.read(reinterpret_cast<char*>(&x), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&y), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&z), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&label), sizeof(float))) break;

        globalPoints.push_back({x,y,z,label});
    }
    fin.close();
    return true;
}

// ------------------- Utility: load cov.txt => clusterInfos -------------------
bool loadCovTxt(const std::string &filename, std::vector<ClusterInfo> &clusterInfos){
    clusterInfos.clear();
    std::ifstream fin(filename);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open cov txt: " << filename << std::endl;
        return false;
    }
    std::string line;
    while(std::getline(fin, line)){
        if(line.find("ClusterIdx:") == std::string::npos){
            continue;
        }
        ClusterInfo cinfo;
        // read 'Original Indices:' lines
        while(true){
            std::streampos pos = fin.tellg();
            if(!std::getline(fin,line)){
                break;
            }
            if(line.find("Original Indices:") != std::string::npos){
                while(std::getline(fin,line)){
                    if(line.empty()) break;
                    size_t idx_val = std::stoul(line);
                    cinfo.original_indices.push_back(idx_val);
                }
                break;
            } else {
                fin.seekg(pos);
                break;
            }
        }
        clusterInfos.push_back(cinfo);
    }
    fin.close();
    return true;
}

// ------------------- Utility: invert SE3 4x4 -------------------
Eigen::Matrix4d inverseSE3(const Eigen::Matrix4d &T){
    // top-left 3x3 is R, top-right 3x1 is t
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    Eigen::Matrix3d Rinv = R.transpose();
    Eigen::Vector3d tinv = - Rinv * t;

    Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
    Tinv.block<3,3>(0,0) = Rinv;
    Tinv.block<3,1>(0,3) = tinv;
    return Tinv;
}

// ------------------- (1) Compute cluster centers & normals -------------------
bool computeClusterCentersAndNormals(const std::vector<GlobalPoint> &globalPoints,
                                     const std::vector<ClusterInfo> &clusterInfos,
                                     std::vector<ClusterCenter> &centers)
{
    centers.clear();
    centers.reserve(clusterInfos.size());
    for(size_t i=0; i<clusterInfos.size(); i++){
        const auto &cinfo = clusterInfos[i];
        size_t N = cinfo.original_indices.size();
        if(N<3){
            // can't compute normal for <3 points
            centers.push_back({0,0,0,0,0,0});
            continue;
        }
        // gather in an Eigen matrix => for centroid + covariance
        Eigen::Matrix<double, 3, Eigen::Dynamic> neighbors(3, N);
        for(size_t j=0; j<N; j++){
            size_t idx = cinfo.original_indices[j];
            if(idx>=globalPoints.size()){
                std::cerr << "[Warning] cluster index out of range\n";
                return false;
            }
            neighbors(0,j) = globalPoints[idx].x;
            neighbors(1,j) = globalPoints[idx].y;
            neighbors(2,j) = globalPoints[idx].z;
        }
        // centroid
        Eigen::Vector3d center = neighbors.rowwise().mean();
        // subtract mean
        for(size_t col=0; col<N; col++){
            neighbors.col(col) -= center;
        }
        // cov
        Eigen::Matrix3d cov = (neighbors * neighbors.transpose()) / double(N);

        // eigen decomposition => normal = evec w/ min eigenvalue
        Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Vector3cd eigVals = es.eigenvalues();
        Eigen::Matrix3cd eigVecs = es.eigenvectors();
        // find min
        double minVal = 1e30; 
        int minIdx = 0;
        for(int k=0; k<3; k++){
            double val = eigVals(k).real();
            if(val < minVal){
                minVal = val;
                minIdx = k;
            }
        }
        // normal
        Eigen::Vector3d normal;
        normal << eigVecs(0,minIdx).real(),
                  eigVecs(1,minIdx).real(),
                  eigVecs(2,minIdx).real();
        normal.normalize();

        ClusterCenter cc;
        cc.x = center(0);
        cc.y = center(1);
        cc.z = center(2);
        cc.nx = normal(0);
        cc.ny = normal(1);
        cc.nz = normal(2);

        centers.push_back(cc);
    }
    return true;
}

// ------------------- main -------------------
int main(int argc, char** argv){
    if(argc < 6){
        std::cerr << "Usage: " << argv[0]
                  << " <pose.txt> <centroids.bin> <cov.txt> <global_map.bin> <output_dir> [calib.txt]\n";
        return 1;
    }

    std::string pose_file   = argv[1];
    std::string centroids_file = argv[2];  // (unused if we do own centroid calc)
    std::string cov_file    = argv[3];
    std::string global_file = argv[4];
    std::string out_dir     = argv[5];
    std::string calib_file  = (argc==7)? argv[6] : "";

    // 1) load poses (camera->map)
    std::vector<Eigen::Matrix4d> poses;
    if(!loadKITTIposes(pose_file, poses)){
        return 1;
    }
    std::cout << "[Info] loaded " << poses.size() << " poses.\n";

    // 2) load global map
    std::vector<GlobalPoint> globalPoints;
    if(!loadGlobalMap(global_file, globalPoints)){
        return 1;
    }
    std::cout << "[Info] global map size: " << globalPoints.size() << std::endl;

    // 3) load cov.txt => clusterInfos
    std::vector<ClusterInfo> clusterInfos;
    if(!loadCovTxt(cov_file, clusterInfos)){
        return 1;
    }
    std::cout << "[Info] clusterInfos size: " << clusterInfos.size() << std::endl;

    // 4) compute cluster centers & normals
    std::vector<ClusterCenter> centers;
    if(!computeClusterCentersAndNormals(globalPoints, clusterInfos, centers)){
        return 1;
    }
    if(centers.size() != clusterInfos.size()){
        std::cerr << "[Error] mismatch centers.size vs clusterInfos\n";
        return 1;
    }
    std::cout << "[Info] computed " << centers.size() << " cluster centers.\n";

    // 5) calibration => T_lidar->camera => T_{cl} => invert => T_{lc}
    Eigen::Matrix4d T_cl = Eigen::Matrix4d::Identity();
    if(!calib_file.empty()){
        std::ifstream cfin(calib_file);
        if(cfin.is_open()){
            for(std::string line; std::getline(cfin,line); ){
                if(line.rfind("Tr:", 0)==0){
                    std::stringstream ss(line.substr(3));
                    std::vector<double> vals; double v;
                    while(ss>>v) vals.push_back(v);
                    if(vals.size()==12){
                        for(int r=0; r<3; r++){
                            for(int c=0; c<4; c++){
                                T_cl(r,c) = vals[r*4 + c];
                            }
                        }
                        T_cl(3,3)=1.0;
                    }
                    break;
                }
            }
            cfin.close();
            std::cout << "[Info] LiDAR->Camera loaded from " << calib_file << "\n";
        }
    }
    // map->lidar => T_lm = T_cl * T_cm
    // but T_cm = inverseSE3(poses[f]) => camera->map => invert => map->camera
    // also T_lc = inverseSE3(T_cl) if needed
    // threshold
    double threshold = 100.0; 

    // create output dir
    {
        std::string cmd = "mkdir -p " + out_dir;
        system(cmd.c_str());
    }

    // 6) for each frame => transform cluster centers => filter => save .bin
    for(size_t f=0; f<poses.size(); f++){
        Eigen::Matrix4d T_cm = poses[f].inverse(); // map->camera
        Eigen::Matrix4d T_lm = T_cl * T_cm;        // map->lidar

        // gather
        std::vector<float> outData;
        for(size_t cidx=0; cidx<centers.size(); cidx++){
            const auto &cc = centers[cidx];
            Eigen::Vector4d c_map(cc.x, cc.y, cc.z, 1.0);
            Eigen::Vector4d c_lidar = T_lm * c_map;
            double dist = c_lidar.head<3>().norm();
            if(dist <= threshold){
                Eigen::Matrix3d R_lm = T_lm.block<3,3>(0,0);
                Eigen::Vector3d n_map(cc.nx, cc.ny, cc.nz);
                Eigen::Vector3d n_lidar = R_lm * n_map; // transform normal
                n_lidar.normalize();

                // push x,y,z,nx,ny,nz => 6 floats
                outData.push_back((float)c_lidar(0));
                outData.push_back((float)c_lidar(1));
                outData.push_back((float)c_lidar(2));
                outData.push_back((float)n_lidar(0));
                outData.push_back((float)n_lidar(1));
                outData.push_back((float)n_lidar(2));
            }
        }

        // save => out_dir/f=000000.bin
        char buf[256];
        snprintf(buf, sizeof(buf), "%06zu.bin", f);
        std::string out_file = out_dir + "/" + buf;

        std::ofstream ofs(out_file, std::ios::binary);
        if(!ofs.is_open()){
            std::cerr << "[Error] can't open " << out_file << "\n";
            continue;
        }
        if(!outData.empty()){
            ofs.write(reinterpret_cast<const char*>(outData.data()),
                      outData.size()*sizeof(float));
        }
        ofs.close();
        std::cout << "[Frame " << f << "] wrote " 
                  << (outData.size()/6) << " clusters => " << out_file << std::endl;
    }

    std::cout << "[Done] All frames processed.\n";
    return 0;
}
