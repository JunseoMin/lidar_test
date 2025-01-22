#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Dense>

// ---------------------------------------------------------
// (A) Data structures
// ---------------------------------------------------------
struct ClusterCentroid {
    float x,y,z; 
};

struct ClusterInfo {
    // 전역 맵에서 이 클러스터가 가지는 point index 목록
    std::vector<size_t> original_indices;
};

struct GlobalPoint {
    float x,y,z,label; 
};

// ---------------------------------------------------------
// (B) Function: Load KITTI pose as camera->map (T_{cm})
// ---------------------------------------------------------
bool loadKITTIposes(const std::string &pose_file, std::vector<Eigen::Matrix4d> &poses)
{
    std::ifstream fin(pose_file);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open pose file: " << pose_file << std::endl;
        return false;
    }
    poses.clear();

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
        // KITTI: 3x4 => camera->map
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

// ---------------------------------------------------------
// (C) Load cluster centroids.bin ([C,3])
// ---------------------------------------------------------
bool loadCentroidsBin(const std::string &filename, std::vector<ClusterCentroid> &centroids)
{
    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open centroids bin: " << filename << std::endl;
        return false;
    }
    centroids.clear();

    while(true){
        float x=0, y=0, z=0;
        fin.read(reinterpret_cast<char*>(&x), sizeof(float));
        if(!fin.good()) break;
        fin.read(reinterpret_cast<char*>(&y), sizeof(float));
        if(!fin.good()) break;
        fin.read(reinterpret_cast<char*>(&z), sizeof(float));
        if(!fin.good()) break;

        centroids.push_back({x,y,z});
    }
    fin.close();
    return true;
}

// ---------------------------------------------------------
// (D) Load cov.txt => clusterInfos
// ---------------------------------------------------------
bool loadCovTxt(const std::string &filename, std::vector<ClusterInfo> &clusterInfos)
{
    std::ifstream fin(filename);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open cov txt: " << filename << std::endl;
        return false;
    }
    clusterInfos.clear();

    std::string line;
    while(std::getline(fin, line)){
        if(line.find("ClusterIdx:") == std::string::npos) {
            continue;
        }
        // read Original Indices
        ClusterInfo cinfo;

        while(true){
            std::streampos pos = fin.tellg();
            if(!std::getline(fin, line)){
                break;
            }
            if(line.find("Original Indices:") != std::string::npos){
                while(std::getline(fin, line)){
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

// ---------------------------------------------------------
// (E) Load global_map.bin => [N,4] float => (x,y,z,label)
// ---------------------------------------------------------
bool loadGlobalMap(const std::string &filename, std::vector<GlobalPoint> &globalPoints)
{
    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open global map: " << filename << std::endl;
        return false;
    }
    globalPoints.clear();

    while(true){
        float x=0,y=0,z=0,label=0;
        fin.read(reinterpret_cast<char*>(&x), sizeof(float));
        if(!fin.good()) break;
        fin.read(reinterpret_cast<char*>(&y), sizeof(float));
        if(!fin.good()) break;
        fin.read(reinterpret_cast<char*>(&z), sizeof(float));
        if(!fin.good()) break;
        fin.read(reinterpret_cast<char*>(&label), sizeof(float));
        if(!fin.good()) break;

        globalPoints.push_back({x,y,z,label});
    }
    fin.close();
    return true;
}

// ---------------------------------------------------------
// (F) Convert invert a 4x4 SE(3) transform
// ---------------------------------------------------------
Eigen::Matrix4d inverseSE3(const Eigen::Matrix4d &T)
{
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);

    Eigen::Matrix3d Rinv = R.transpose();
    Eigen::Vector3d tinv = - Rinv * t;

    Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
    Tinv.block<3,3>(0,0) = Rinv;
    Tinv.block<3,1>(0,3) = tinv;
    return Tinv;
}

// ---------------------------------------------------------
// (G) main
// ---------------------------------------------------------
int main(int argc, char** argv)
{
    if(argc < 6){
        std::cerr << "Usage: " << argv[0]
                  << " <pose.txt> <centroids.bin> <cov.txt> <global_map.bin> <output_dir> [calib.txt]\n";
        return 1;
    }

    std::string pose_file      = argv[1]; // camera->map
    std::string centroids_file = argv[2];
    std::string cov_file       = argv[3];
    std::string global_map_bin = argv[4];
    std::string output_dir     = argv[5];
    std::string calib_file     = (argc==7)? argv[6]:"";

    // 1) load poses => T_{cm} (camera->map)
    std::vector<Eigen::Matrix4d> poses;
    if(!loadKITTIposes(pose_file, poses)){
        return 1;
    }
    std::cout << "[Info] loaded " << poses.size() << " camera->map poses.\n";

    // 2) load cluster centroids + info
    std::vector<ClusterCentroid> centroids;
    if(!loadCentroidsBin(centroids_file, centroids)) {
        return 1;
    }
    // std::vector<ClusterInfo> clusterInfos;
    // if(!loadCovTxt(cov_file, clusterInfos)){
    //     return 1;
    // }
    // if(clusterInfos.size() != centroids.size()){
    //     std::cerr << "[Warning] clusterInfos.size(" << clusterInfos.size()
    //               << ") != centroids.size(" << centroids.size() << ")\n";
    // }

    // 3) load global map => [N,4] float => (x,y,z,label)
    // std::vector<GlobalPoint> globalPoints;
    // if(!loadGlobalMap(global_map_bin, globalPoints)){
    //     return 1;
    // }
    // std::cout << "[Info] global map points: " << globalPoints.size() << std::endl;

    // 4) load calibration => T_{lc} (LiDAR->camera)
    //    => T_{cl} = inverseSE3(T_{lc})
    Eigen::Matrix4d T_cl = Eigen::Matrix4d::Identity();
    bool use_calib = false;
    if(!calib_file.empty()){
        std::ifstream cfin(calib_file);
        if(cfin.is_open()){
            use_calib = true;
            for(std::string line; std::getline(cfin,line);){
                if(line.rfind("Tr:",0)==0){
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
            std::cout << "[Info] LiDAR->Camera loaded:\n" << T_cl << std::endl;
        } else {
            std::cerr << "[Warning] can't open " << calib_file << ", skip.\n";
        }
    }

    // => T_{cl} = (T_{lc})^-1
    Eigen::Matrix4d T_lc = inverseSE3(T_cl);

    // => for each frame, we want to measure distance in "LiDAR coords"
    // The transform from "map" -> "lidar" is T_{lm} = T_cl * T_{mc} 
    // but T_{mc} = inverseSE3( T_{cm} ) => so T_{lm} = T_cl * T_{mc} 
    // Then c_lidar = T_{lm} * c_map
    double threshold = 100.0; // 100m

    // create output dir
    {
        std::string cmd = "mkdir -p " + output_dir;
        system(cmd.c_str());
    }

    // process each frame
    for(size_t f=0; f < poses.size(); f++)
    {
        const Eigen::Matrix4d &T_mc = poses[f]; // camera->map
        Eigen::Matrix4d T_cm = inverseSE3(T_mc); // map->camera
        Eigen::Matrix4d T_lm = T_lc * T_cm;       // map->lidar
        std::vector<float> outPoints;

        // std::vector<size_t> selected_clusters;
        // selected_clusters.reserve(centroids.size());

        // 1) centroid->lidar => distance check
        for(size_t cidx=0; cidx<centroids.size(); cidx++){
            float mx = centroids[cidx].x; 
            float my = centroids[cidx].y; 
            float mz = centroids[cidx].z;

            // homogeneous
            Eigen::Vector4d c_map(mx, my, mz, 1.0);
            Eigen::Vector4d c_lidar = T_lm * c_map; 
            
            double dist_lidar = std::sqrt(c_lidar(0)*c_lidar(0)
                                        + c_lidar(1)*c_lidar(1)
                                        + c_lidar(2)*c_lidar(2));
            if(dist_lidar <= threshold){
                // selected_clusters.push_back(cidx);
                outPoints.push_back((float)c_lidar(0));
                outPoints.push_back((float)c_lidar(1));
                outPoints.push_back((float)c_lidar(2));
            }
        }

        // 2) gather points => transform "map->lidar" => store as [x,y,z,label]
        // for(size_t idx : selected_clusters){
        //     if(idx >= clusterInfos.size()) continue;
        //     const auto &cinfo = clusterInfos[idx];
        //     // original_indices
        //     for(auto origIdx : cinfo.original_indices){
        //         if(origIdx < globalPoints.size()){
        //             float mx = globalPoints[origIdx].x;
        //             float my = globalPoints[origIdx].y;
        //             float mz = globalPoints[origIdx].z;
        //             float label = globalPoints[origIdx].label;

        //             Eigen::Vector4d p_map(mx, my, mz, 1.0);
        //             Eigen::Vector4d p_lidar = T_lm * p_map;

        //             outPoints.push_back((float)p_lidar(0));
        //             outPoints.push_back((float)p_lidar(1));
        //             outPoints.push_back((float)p_lidar(2));
        //             // outPoints.push_back(label);
        //         }
        //     }
        // }

        // 3) save => 000000.bin ...
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << f << ".bin";
        std::string out_file = output_dir + "/" + oss.str();

        std::ofstream ofs(out_file, std::ios::binary);
        if(!ofs.is_open()){
            std::cerr << "[Error] cannot open " << out_file << "\n";
            continue;
        }
        if(!outPoints.empty()){
            ofs.write(reinterpret_cast<const char*>(outPoints.data()),
                      outPoints.size()*sizeof(float));
        }
        ofs.close();

        if (outPoints.size()/4 == 0){
            std::cout << "[Frame " << f << "] total points: " << (outPoints.size()/4)
                      << " => " << out_file << std::endl;
        }
    }

    std::cout << "[Done] All frames processed!\n";
    return 0;
}
