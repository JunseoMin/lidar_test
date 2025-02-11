#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>   // for system()

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
};

// ------------------- Utility: create directories if needed (Linux/mac) -------------------
bool createDirectoriesIfNeeded(const std::string &file_path)
{
    // 파일 경로에서 디렉토리 경로만 추출
    // 예) file_path = "map/seq/00.bin" => dir_path = "map/seq"
    auto pos = file_path.rfind('/');
    if(pos == std::string::npos) {
        // 슬래시가 없다면 현재 디렉토리에 저장한다고 가정
        // 특별히 만들 디렉토리가 없으므로 바로 true 반환
        return true;
    }

    std::string dir_path = file_path.substr(0, pos);

    // system()을 통해 "mkdir -p dir_path" 실행 (리눅스/맥 전용)
    std::string cmd = "mkdir -p \"" + dir_path + "\"";
    int ret = system(cmd.c_str());
    if(ret != 0) {
        std::cerr << "[Error] Failed to create directory: " << dir_path << std::endl;
        return false;
    }
    return true;
}

// ------------------- Utility: load global_map.bin => [x,y,z,label] float -------------------
bool loadGlobalMap(const std::string &filename, std::vector<GlobalPoint> &globalPoints) {
    globalPoints.clear();
    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "[Error] cannot open global map: " << filename << std::endl;
        return false;
    }
    while(true){
        float x=0, y=0, z=0, label=0;
        if(!fin.read(reinterpret_cast<char*>(&x), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&y), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&z), sizeof(float))) break;
        if(!fin.read(reinterpret_cast<char*>(&label), sizeof(float))) break;

        globalPoints.push_back({x, y, z, label});
    }
    fin.close();
    return true;
}

// ------------------- Utility: load cov.txt => clusterInfos -------------------
bool loadCovTxt(const std::string &filename, std::vector<ClusterInfo> &clusterInfos) {
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
        if(N < 3){
            // can't compute normal for < 3 points
            centers.push_back({0,0,0, 0,0,0});
            continue;
        }
        // gather in an Eigen matrix => for centroid + covariance
        Eigen::Matrix<double, 3, Eigen::Dynamic> neighbors(3, N);
        for(size_t j=0; j<N; j++){
            size_t idx = cinfo.original_indices[j];
            if(idx >= globalPoints.size()){
                std::cerr << "[Warning] cluster index out of range: " << idx << "\n";
                return false;
            }
            neighbors(0, j) = globalPoints[idx].x;
            neighbors(1, j) = globalPoints[idx].y;
            neighbors(2, j) = globalPoints[idx].z;
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

        // find min eigenvalue => normal
        double minVal = 1e30; 
        int minIdx = 0;
        for(int k=0; k<3; k++){
            double val = eigVals(k).real();
            if(val < minVal){
                minVal = val;
                minIdx = k;
            }
        }
        Eigen::Vector3d normal;
        normal << eigVecs(0, minIdx).real(),
                  eigVecs(1, minIdx).real(),
                  eigVecs(2, minIdx).real();
        normal.normalize();

        ClusterCenter cc;
        cc.x = (float)center(0);
        cc.y = (float)center(1);
        cc.z = (float)center(2);
        cc.nx = (float)normal(0);
        cc.ny = (float)normal(1);
        cc.nz = (float)normal(2);

        centers.push_back(cc);
    }
    return true;
}

// ------------------- main -------------------
int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "Usage: " << argv[0]
                  << " <cov.txt> <global_map.bin> <output_xyzn.bin>\n";
        return 1;
    }

    // 1) 파일 입력 경로 설정
    std::string cov_file     = argv[1];
    std::string global_file  = argv[2];
    std::string out_file     = argv[3];

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

    // 4) compute cluster centers & normals (이미 map 좌표계 기준)
    std::vector<ClusterCenter> centers;
    if(!computeClusterCentersAndNormals(globalPoints, clusterInfos, centers)){
        return 1;
    }
    std::cout << "[Info] computed " << centers.size() << " cluster centers.\n";

    // 5) 필요한 디렉토리가 없으면 생성
    if(!createDirectoriesIfNeeded(out_file)){
        std::cerr << "[Error] Failed to create necessary directories for output.\n";
        return 1;
    }

    // 6) .bin 파일로 (x, y, z, nx, ny, nz) 저장
    {
        std::ofstream ofs(out_file, std::ios::binary);
        if(!ofs.is_open()){
            std::cerr << "[Error] cannot open " << out_file << std::endl;
            return 1;
        }
        // 순서: x, y, z, nx, ny, nz (float 6개)
        for(const auto& cc : centers){
            ofs.write(reinterpret_cast<const char*>(&cc.x),  sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&cc.y),  sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&cc.z),  sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&cc.nx), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&cc.ny), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&cc.nz), sizeof(float));
        }
        ofs.close();
        std::cout << "[Info] Saved " << centers.size()
                  << " xyzn to " << out_file << std::endl;
    }

    std::cout << "[Done] Map-level xyznormal extraction complete.\n";
    return 0;
}
