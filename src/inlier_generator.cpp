#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <unordered_set>
#include "semantic_teaser.h"
#include "cluster_manager.h"  // 가정: 여기 안에 cluster 정보가 있음

// 예시 PointType
using PointType = pcl::PointXYZ;

// 가정 1) semanticTeaser::Correspondences 에서 map 쪽 인덱스: 
//         initial_correspondences.indices.second[colIndex]
// 가정 2) clusterManager 안에 "std::vector<ClusterInfo> clusters_"와
//         "std::vector<int> pointIdToClusterId" 등이 존재한다고 가정
//         (pointIdToClusterId[ptIdx] -> cluster ID)

void saveInlierMapClusters(
    semanticTeaser& solver, 
    clusterManager& map_manager, 
    pcl::PointCloud<PointType>::Ptr mapCloud, 
    const std::string& out_pcd_filename)
{
    // 1) Inlier 인덱스(최대 클리크) 가져오기
    const std::vector<int>& inliers = solver.getInlierMaxClique();
    if (inliers.empty()) {
        std::cout << "[Warning] No inliers found!\n";
        return;
    }

    // 2) inlier가 가리키는 맵 점들의 cluster ID를 모을 컨테이너
    //    (중복 방지 위해 set or unordered_set 사용)
    std::unordered_set<int> inlierClusterIDs;

    // 보통 solver 내부에 저장된 correspondences 접근
    // semanticTeaser::Correspondences& corrs = solver.initial_correspondences; 
    // 위처럼 직접 접근할 수도 있으나, 
    // 접근제어(public/private)에 따라 getter 함수를 만들 수도 있음.
    const auto& corrs = solver.initial_correspondences; // 가정: 구현 필요

    // 3) inlier 중 map 측 인덱스 추출 & cluster ID 찾기
    for (int inlierIdx : inliers) {
        // correspondences.indices.second[inlierIdx] = map 점의 인덱스
        int map_point_idx = corrs.indices.second[inlierIdx];

        // map_point_idx -> cluster ID
        // clusterManager에서 pointToClusterId를 관리한다고 가정
        // (실제로는 clusterManager의 구현에 맞춰 수정)
        int cluster_id = map_manager.pointIdToClusterId[map_point_idx];

        inlierClusterIDs.insert(cluster_id);
    }

    // 4) inlier인 cluster들만 모아서 하나의 PCL로 생성
    pcl::PointCloud<PointType>::Ptr inlierClusterCloud(new pcl::PointCloud<PointType>);

    // clusterManager에서 clusters_ : 
    //   vector< pcl::PointCloud<PointType>::Ptr > clusters_;
    // 혹은 vector< vector<int> > 로 각 클러스터 점 인덱스를 관리할 수도 있음.
    // 여기서는 'clusters_'가 (index 배열)인지 (cloud 포인터)인지 예시에 맞춰 작성:
    for (int cid : inlierClusterIDs) {
        // 예: map_manager.clusters_[cid].indices -> 그 클러스터에 속한 점들의 인덱스
        const auto& idx_vec = map_manager.clusters_[cid].indices; 
        for (int ptIdx : idx_vec) {
            inlierClusterCloud->push_back(mapCloud->points[ptIdx]);
        }
    }

    inlierClusterCloud->width = inlierClusterCloud->size();
    inlierClusterCloud->height = 1;
    inlierClusterCloud->is_dense = true;

    // 5) inlier cluster PCD 저장
    if (!inlierClusterCloud->empty()) {
        pcl::io::savePCDFileBinary(out_pcd_filename, *inlierClusterCloud);
        std::cout << "[Info] Saved " << inlierClusterCloud->size() 
                  << " inlier-cluster points to: " << out_pcd_filename << "\n";
    } else {
        std::cout << "[Warning] No inlier cluster points to save!\n";
    }
}

// -------------------------------------------------------
// 실제 사용 예시
int main()
{
    // 0) solver, map_manager, map point cloud 준비
    //    (자세한 것은 기존 프로젝트 구조를 참고)
    semanticTeaser solver( /*...*/ );
    clusterManager map_manager;
    pcl::PointCloud<PointType>::Ptr mapCloud(new pcl::PointCloud<PointType>);

    // ... 여기에 TEASER solve 과정, clusterManager로 맵 클러스터 구성 등등 ...

    // 1) TEASER로 solve
    // solver.solve(...);

    // 2) 저장
    saveInlierMapClusters(solver, map_manager, mapCloud, "inlier_map_clusters.pcd");

    return 0;
}
