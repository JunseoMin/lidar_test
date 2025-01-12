cmake_minimum_required(VERSION 3.10)
project(ProcessPointCloud)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake")

# Find PCL
find_package(PCL REQUIRED COMPONENTS common io kdtree filters segmentation)
set(PCL_LIBRARIES pcl_common pcl_io pcl_kdtree pcl_filters pcl_segmentation)

# Include directories
include_directories("/usr/include/pcl-1.12")
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the executable
add_executable(process_point_cloud mapgeneration.cpp)
target_link_libraries(process_point_cloud
    ${PCL_LIBRARIES}
    pcl_common
    pcl_io
    pcl_kdtree
    pcl_filters
    pcl_segmentation
    pcl_search
    OpenMP::OpenMP_CXX
)

message(STATUS "PCL Libraries: ${PCL_LIBRARIES}")

# Output binary location
set_target_properties(process_point_cloud PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
