cmake_minimum_required(VERSION 3.10)
project(ClusterExtractionByPose)

set(CMAKE_CXX_STANDARD 14) # 또는 14

# PCL 패키지 찾기
find_package(PCL REQUIRED COMPONENTS common io kdtree octree filters segmentation)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# Include 디렉토리 추가
include_directories("/usr/include/pcl-1.12")
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 실행 파일 및 소스 파일 정의
# add_executable(inlier_gt_generator
#     inlier_gt_generator.cpp
# )

add_executable(downsample_voxel
    points_downsample.cpp
)

# PCL 라이브러리 연결
# target_link_libraries(inlier_gt_generator ${PCL_LIBRARIES})
target_link_libraries(downsample_voxel 
    ${PCL_LIBRARIES}
    pcl_common
    pcl_io
    pcl_kdtree
    pcl_filters
    pcl_segmentation
    pcl_search
)

# OpenMP 설정
if (OPENMP_FOUND)
    target_compile_options(downsample_voxel PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(downsample_voxel OpenMP::OpenMP_CXX)
endif()
