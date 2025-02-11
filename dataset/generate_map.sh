#!/bin/bash
echo "Running inlier_gt_generator..."

COV_BASE="/home/server01/js_ws/dataset/encoder_dataset_TEASER/inlier_map"
GT_MAP_BASE="/home/server01/js_ws/dataset/encoder_dataset_TEASER/semantic_map"
OUTPUT_BASE="/home/server01/js_ws/dataset/encoder_dataset_TEASER/map"

# generate GT datset FOR tran
for i in $(seq -w 0 10); do
    echo "Running for sequence $i"
    /home/server01/js_ws/lidar_test/dataset/build/generate_centroid_xyzn_map \
        "$COV_BASE/$i/cov.txt" \
        "$GT_MAP_BASE/${i}_map.bin" \
        "$OUTPUT_BASE/$i/${i}xyzn_map.bin"
done

echo "All processed."
