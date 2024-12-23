#ifndef _INCLUDE_DATA_IO_HPP
#define _INCLUDE_DATA_IO_HPP

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// boost
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

// Eigen
#include <Eigen/Core>

// #include <glog/logging.h>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

bool save_problematic_frames(std::ofstream &out, int src_index, int tgt_index) {
    if (!out) {
        return false;
    }

    std::string space_delimiter = " ";
    out << src_index << space_delimiter << tgt_index << std::endl;

    return true;
}

bool save_timing(std::ofstream &out,
                 std::vector<double> time_vec,
                 int src_index,
                 int tgt_index) {
    if (!out) {
        return false;
    }
    std::string space_delimiter = " ";

    out << src_index << space_delimiter << tgt_index << space_delimiter;

    for (size_t i = 0; i < time_vec.size(); ++i) {
        if (i == time_vec.size() - 1) {
            out << time_vec[i];
        } else {
            out << time_vec[i] << space_delimiter;
        }
    }
    out << std::endl;

    return true;
}

bool save_result(std::ofstream &out,
                 std::vector<double> result_vec,
                 int index) {
    if (!out) {
        return false;
    }
    std::string space_delimiter = " ";

    out << index << space_delimiter;

    for (size_t i = 0; i < result_vec.size(); ++i) {
        if (i == result_vec.size() - 1) {
            out << result_vec[i];
        } else {
            out << result_vec[i] << space_delimiter;
        }
    }
    out << std::endl;

    return true;
}

bool save_pose_error(std::ofstream &out,
                     std::vector<double> e_t,
                     std::vector<double> e_r,
                     int src_index,
                     int tgt_index) {
    if (!out) {
        return false;
    }
    if (e_t.size() != e_r.size()) {
        return false;
    }
    std::string space_delimiter = " ";

    out << src_index << space_delimiter << tgt_index << space_delimiter;

    for (size_t i = 0; i < e_t.size(); ++i) {
        if (i == e_t.size() - 1) {
            out << e_t[i] << space_delimiter << e_r[i];
        } else {
            out << e_t[i] << space_delimiter << e_r[i] << space_delimiter;
        }
    }
    out << std::endl;

    return true;
}

bool read_index_list(std::string index_file_path,
                     std::vector<int> &source_indx,
                     std::vector<int> &target_indx) {
    std::ifstream in(index_file_path, std::ios::in);

    if (!in) {
        return 0;
    }

    while (!in.eof()) {
        int p1, p2;
        in >> p1 >> p2;

        if (in.fail()) {
            break;
        }
        source_indx.emplace_back(p1);
        target_indx.emplace_back(p2);
    }

    return 1;
}

/**
 * @brief save precomputed covariance matrix to one binary file
 *
 * @param fout out stream
 * @param matrix_vector covariance matrix to be saved
 * @return true
 * @return false
 */
bool save_covariance_vec(
        std::ofstream &fout,
        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
                &matrix_vector) {
    if (!fout.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return false;
    }
    size_t vector_size = matrix_vector.size();
    fout.write(reinterpret_cast<const char *>(&vector_size), sizeof(size_t));

    for (const auto &matrix : matrix_vector) {
        fout.write(reinterpret_cast<const char *>(matrix.data()),
                   sizeof(Eigen::Matrix3d));
    }
    fout.close();
    return true;
}

/**
 * @brief read the covariance vector of a point cloud from the file
 *
 * @param fin input stream of the file been opened
 * @param matrix_vector covariance vector to be stored
 * @return true
 * @return false
 */
bool read_covariance_vec(
        std::ifstream &fin,
        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
                &matrix_vector) {
    if (!fin.is_open()) {
        std::cerr << "Failed to open file for reading." << std::endl;
        return false;
    }

    size_t vector_size;
    fin.read(reinterpret_cast<char *>(&vector_size), sizeof(size_t));

    for (size_t i = 0; i < vector_size; ++i) {
        Eigen::Matrix3d mat;
        fin.read(reinterpret_cast<char *>(mat.data()),
                 sizeof(double) * mat.size());
        matrix_vector.push_back(mat);
    }

    fin.close();
    return true;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
load_poses_from_transform_matrix(const std::string filepath) {
    double tmp[12];
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
            pose_vec;
    Eigen::Matrix4d temp_pose = Eigen::Matrix4d::Identity();
    std::ifstream posereader(filepath);

    int count = 0;
    while (posereader >> tmp[0]) {
        for (int i = 1; i < 12; ++i) {
            posereader >> tmp[i];
        }
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp_pose(j, k) = tmp[4 * j + k];
            }
        }

        pose_vec.push_back(temp_pose);

        count++;
        // LOG(WARNING) << temp_pose;
    }
    return pose_vec;
}

/**
 * @brief Convert vectorized calib mat to matrix form
 * 
 * @param matrix_data vectorized 1*12 calib matrix
 * @param calib_mat 4*4 calib matrix
 * @return true 
 * @return false 
 */
bool vec2calib(const std::vector<double> matrix_data, Eigen::Matrix4d &calib_mat) {
    if(matrix_data.size() != 12)
    {
        std::cerr << "[Read Calib Error!] Calibration matrix must have 12 elements." << std::endl;
        return false;
    }

    calib_mat.setIdentity();

    int index = 0;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            calib_mat(i, j) = matrix_data[index++];
        }
    }
    return true;
}

bool load_calib_mat(const std::string calib_file, Eigen::Matrix4d &calib_mat) {
    calib_mat.setIdentity();

    std::ifstream calibreader(calib_file);
    // cout<<filepath<<"\n";
    if (calibreader.is_open()) {
        while (calibreader.peek() != EOF) {
            std::string line;
            getline(calibreader, line);

            std::stringstream ss(line);

            std::string flag;

            ss >> flag;

            if (flag.compare("Tr:") == 0) {
                ss >> calib_mat(0, 0) >> calib_mat(0, 1) >> calib_mat(0, 2) >>
                        calib_mat(0, 3) >> calib_mat(1, 0) >> calib_mat(1, 1) >>
                        calib_mat(1, 2) >> calib_mat(1, 3) >> calib_mat(2, 0) >>
                        calib_mat(2, 1) >> calib_mat(2, 2) >> calib_mat(2, 3);
                break;
            }
        }
        calibreader.close();
    } else
        return 0;

    // LOG(INFO) << "Calib matrix loaded\n";

    // std::cout << setprecision(16) << calib_mat << std::endl;

    return 1;
}

std::string kitti_zero_padding(int &file_name) {
    std::ostringstream ss;
    ss << std::setw(6) << std::setfill('0') << file_name;
    std::string s2(ss.str());
    return s2;
}

bool batch_read_filenames_in_folder(const std::string &folderName,
                                    const std::string &file_list_extenstion,
                                    const std::string &extension,
                                    std::vector<std::string> &fileNames,
                                    int frame_begin = 0,
                                    int frame_end = 99999,
                                    int frame_step = 1) {
    std::string filename_list = folderName + file_list_extenstion;

    // read image filename
    std::ifstream name_list_file(filename_list.c_str(), std::ios::in);
    if (!name_list_file.is_open()) {
        // LOG(WARNING) << "open filename_list failed, file is: " <<
        // filename_list;
        return 0;
    }

    int frame_count = 0;

    while (name_list_file.peek() != EOF) {
        std::string cur_file;
        name_list_file >> cur_file;

        // if (!cur_file.empty() &&
        // !cur_file.substr(cur_file.rfind('.')).compare(extension))
        if (!cur_file.empty()) {
            if (frame_count >= frame_begin && frame_count <= frame_end &&
                ((frame_count - frame_begin) % frame_step == 0)) {
                cur_file = folderName + "/" + cur_file;
                fileNames.push_back(cur_file);
                // LOG(INFO) << "Record the file: [" << cur_file << "].";
            }
            frame_count++;
        }
    }
    name_list_file.close();

    return 1;
}

bool write_pcd_file(
        const std::string &fileName,
        const typename pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud,
        bool as_binary = true) {
    // do the reshaping
    pointCloud->width = 1;
    pointCloud->height = pointCloud->points.size();

    if (as_binary) {
        if (pcl::io::savePCDFileBinary(fileName, *pointCloud) == -1) {
            PCL_ERROR("Couldn't write file\n");
            return false;
        }
    } else {
        if (pcl::io::savePCDFile(fileName, *pointCloud) == -1) {
            PCL_ERROR("Couldn't write file\n");
            return false;
        }
    }
    return true;
}

bool writeLiDARPoseAppend(Eigen::Matrix4f &Trans1_2, std::string &output_file) {
    std::ofstream out(output_file, std::ios::app);  // add after the file

    if (!out) return 0;

    std::string delimiter_str = " ";  // or "\t"
    out << setprecision(8);
    out << Trans1_2(0, 0) << delimiter_str << Trans1_2(0, 1) << delimiter_str
        << Trans1_2(0, 2) << delimiter_str << Trans1_2(0, 3) << delimiter_str
        << Trans1_2(1, 0) << delimiter_str << Trans1_2(1, 1) << delimiter_str
        << Trans1_2(1, 2) << delimiter_str << Trans1_2(1, 3) << delimiter_str
        << Trans1_2(2, 0) << delimiter_str << Trans1_2(2, 1) << delimiter_str
        << Trans1_2(2, 2) << delimiter_str << Trans1_2(2, 3) << "\n";
    out.close();
    return 1;
}

#endif  // _INCLUDE_DATA_IO_HPP