#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// 포인트 타입: (x, y, z)
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

/**
 * @brief Load a binary file containing float x,y,z (3 floats per point).
 * @param input_file .bin file path
 * @return PointCloudT::Ptr loaded cloud
 */
PointCloudT::Ptr loadBinFile(const std::string &input_file) {
    std::ifstream infile(input_file, std::ios::binary);
    if(!infile.is_open()){
        std::cerr << "[Error] cannot open " << input_file << std::endl;
        return nullptr;
    }

    // Read the entire file into a buffer
    infile.seekg(0, std::ios::end);
    std::streamsize fileSize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    // Each point has 3 floats => 3 * sizeof(float) = 12 bytes
    // number of points = fileSize / 12
    if (fileSize % 12 != 0) {
        std::cerr << "[Warning] file size not multiple of 12. Possibly corrupted or 4D data.\n";
        // We'll still try reading in multiples of 12
    }
    size_t numPoints = static_cast<size_t>(fileSize / 12);

    std::vector<float> buffer(numPoints * 3);
    if(!infile.read(reinterpret_cast<char*>(buffer.data()), fileSize)){
        std::cerr << "[Error] file read incomplete: " << input_file << std::endl;
        return nullptr;
    }
    infile.close();

    auto cloud = std::make_shared<PointCloudT>();
    cloud->resize(numPoints);

    for(size_t i=0; i<numPoints; i++){
        (*cloud)[i].x = buffer[i*3 + 0];
        (*cloud)[i].y = buffer[i*3 + 1];
        (*cloud)[i].z = buffer[i*3 + 2];
    }
    return cloud;
}

/**
 * @brief Save point cloud as .bin (x,y,z) float
 */
bool saveBinFile(const PointCloudT::Ptr &cloud, const std::string &output_file) {
    std::ofstream ofs(output_file, std::ios::binary);
    if(!ofs.is_open()){
        std::cerr << "[Error] cannot open " << output_file << " for writing.\n";
        return false;
    }
    for(const auto &pt : cloud->points){
        ofs.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
    }
    ofs.close();
    return true;
}

/**
 * @brief Voxel downsample a point cloud with given leaf_size
 */
PointCloudT::Ptr voxelDownsample(const PointCloudT::Ptr &input, float leaf_size){
    PointCloudT::Ptr output(new PointCloudT());
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(input);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*output);
    return output;
}

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "Usage: " << argv[0]
                  << " <input_dir> <output_dir> <leaf_size>\n";
        return 1;
    }

    std::string input_dir  = argv[1];
    std::string output_dir = argv[2];
    float leaf_size        = std::stof(argv[3]);

    std::cout << "[Info] input_dir  = " << input_dir << std::endl;
    std::cout << "[Info] output_dir = " << output_dir << std::endl;
    std::cout << "[Info] leaf_size  = " << leaf_size << std::endl;

    // 1) .bin 파일 전체 수집
    std::vector<std::string> bin_files;
    for(const auto &entry : fs::recursive_directory_iterator(input_dir)){
        if(entry.is_regular_file() && entry.path().extension() == ".bin"){
            bin_files.push_back(entry.path().string());
        }
    }
    // 2) 사전순 정렬 (파일 이름이 zero-padding이면 시간순과 같을 것)
    std::sort(bin_files.begin(), bin_files.end());

    // 3) 병렬 처리(OpenMP)
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<(int)bin_files.size(); i++){
        const auto &bin_path = bin_files[i];

        fs::path rel_path = fs::relative(fs::path(bin_path), fs::path(input_dir));
        fs::path out_path = fs::path(output_dir) / rel_path; // output dir + same relative structure

        fs::create_directories(out_path.parent_path());

        PointCloudT::Ptr cloud = loadBinFile(bin_path);
        if(!cloud || cloud->empty()){
            std::cerr << "[Warning] skip empty or failed cloud: " << bin_path << std::endl;
            saveBinFile(cloud, out_path.string());
            continue;
        }

        PointCloudT::Ptr ds_cloud = voxelDownsample(cloud, leaf_size);

        if(!saveBinFile(ds_cloud, out_path.string())){
            std::cerr << "[Error] fail to save " << out_path << std::endl;
        } else {
            #pragma omp critical
            {
                std::cout << "[Done] " << bin_path << " -> "
                          << out_path.string() << " ("
                          << ds_cloud->size() << " pts)\n";
            }
        }
    }

    std::cout << "[Info] All done. Processed " << bin_files.size() << " bin files.\n";

    return 0;
}
