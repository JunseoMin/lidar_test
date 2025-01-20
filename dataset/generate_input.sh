#!/bin/bash

# echo "Running process_point_cloud..."
# python3 /home/server01/js_ws/lidar_test/dataset/generate_gt_map.py

# # save covariance and centroids of clusters
# echo "Running process_point_cloud..."
# /home/server01/js_ws/lidar_test/reconstruction/build/bin/process_point_cloud "a" "a" "a"

# 명령어를 반복 실행하는 스크립트
echo "Running inlier_gt_generator..."
# 기본 경로 설정
POSES_PATH="/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences"
INLIER_CENTROIDS_BASE="/home/server01/js_ws/dataset/odometry_dataset/inlier_map"
COV_BASE="/home/server01/js_ws/dataset/odometry_dataset/inlier_map"
GT_MAP_BASE="/home/server01/js_ws/dataset/odometry_dataset/semantic_map"
OUTPUT_BASE="/home/server01/js_ws/dataset/odometry_dataset/encoding_gt"
CALIB_BASE="/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences"

# generate GT datset FOR tran
for i in $(seq -w 0 10); do
    echo "Running for sequence $i"
    /home/server01/js_ws/lidar_test/dataset/build/inlier_gt_generator \
        "$POSES_PATH/$i/poses.txt" \
        "$INLIER_CENTROIDS_BASE/$i/centroids.bin" \
        "$COV_BASE/$i/cov.txt" \
        "$GT_MAP_BASE/${i}_map.bin" \
        "$OUTPUT_BASE/$i" \
        "$CALIB_BASE/$i/calib.txt"
done

echo "All processed. Train start!"

# torchrun --nproc_per_node=2 /home/server01/js_ws/lidar_test/train_diffusion_multi.py
