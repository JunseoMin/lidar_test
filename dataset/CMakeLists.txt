cmake_minimum_required(VERSION 3.10)
project(ClusterExtractionByPose)

set(CMAKE_CXX_STANDARD 14)
# or 17

find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(inlier_gt_generator
    inlier_gt_generator.cpp
)

target_link_libraries(inlier_gt_generator
    ${PCL_LIBRARIES}
)

if(OPENMP_FOUND)
    target_compile_options(inlier_gt_generator PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(inlier_gt_generator ${OpenMP_CXX_FLAGS})
endif()
