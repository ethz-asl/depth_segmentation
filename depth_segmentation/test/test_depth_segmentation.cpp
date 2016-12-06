#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>

#include "depth_segmentation/common.h"
#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/testing_entrypoint.h"

namespace depth_segmentation {
class DepthSegmentationTest : public ::testing::Test {
 protected:
  DepthSegmentationTest()
      : depth_camera_(),
        surface_normal_params_(),
        max_distance_map_params_(),
        min_concavity_map_params_(),
        final_edge_map_params_(),
        label_map_params_(),
        depth_segmenter_(depth_camera_, surface_normal_params_,
                         max_distance_map_params_, min_concavity_map_params_,
                         final_edge_map_params_, label_map_params_) {
    cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_32FC1);
    camera_matrix.at<float>(0, 0) = 574.0527954101562f;
    camera_matrix.at<float>(0, 2) = 314.5f;
    camera_matrix.at<float>(1, 1) = 574.0527954101562f;
    camera_matrix.at<float>(1, 2) = 235.5f;
    camera_matrix.at<float>(2, 2) = 1.0f;
    depth_camera_.setCameraMatrix(camera_matrix);
    min_concavity_map_params_.window_size = 3;
  }
  virtual ~DepthSegmentationTest() {}
  virtual void SetUp() {}

  SurfaceNormalParams surface_normal_params_;
  MaxDistanceMapParams max_distance_map_params_;
  MinConcavityMapParams min_concavity_map_params_;
  FinalEdgeMapParams final_edge_map_params_;
  LabelMapParams label_map_params_;
  DepthCamera depth_camera_;
  DepthSegmenter depth_segmenter_;
};

TEST_F(DepthSegmentationTest, testConvexity) {
  static constexpr size_t kNormalImageWidth = 500u;
  cv::Size image_size(kNormalImageWidth, kNormalImageWidth);
  cv::Mat concave_normals(image_size, CV_32FC3);
  cv::Mat depth_image(image_size, CV_32FC1);
  cv::Vec3f x_normal;
  x_normal[0] = 1.0f;
  x_normal[1] = 0.0f;
  x_normal[2] = 0.0f;
  cv::Vec3b y_normal;
  y_normal[0] = 0.0f;
  y_normal[1] = 1.0f;
  y_normal[2] = 0.0f;
  cv::Vec3b z_normal;
  z_normal[0] = 0.0f;
  z_normal[1] = 0.0f;
  z_normal[2] = 1.0f;

  // We are expecting the concavities at half the image width and on the right
  // image side at half image height.
  cv::Mat expected_convexity = cv::Mat::ones(image_size, CV_32FC1);

  for (size_t i = 0u; i < kNormalImageWidth; ++i) {
    for (size_t j = 0u; j < kNormalImageWidth; ++j) {
      if (j == kNormalImageWidth / 2u) {
        expected_convexity.at<float>(i, j) = 0.0f;
      }
      if (j < kNormalImageWidth / 2u) {
        concave_normals.at<cv::Vec3f>(i, j) = x_normal;
        depth_image.at<float>(i, j) = j * 0.01f + 0.2f;
      } else {
        if (i == kNormalImageWidth / 2u) {
          expected_convexity.at<float>(i, j) = 0.0f;
        }
        if (i < kNormalImageWidth / 2u) {
          concave_normals.at<cv::Vec3f>(i, j) = y_normal;
          depth_image.at<float>(i, j) = i * 0.01f + 0.2f;
        } else {
          concave_normals.at<cv::Vec3f>(i, j) = z_normal;
          depth_image.at<float>(i, j) = (kNormalImageWidth / 2u) * 0.01f + 0.2f;
        }
      }
    }
  }
  cv::Mat depth_map(image_size, CV_32FC3);
  depth_segmenter_.computeDepthMap(depth_image, &depth_map);

  cv::Mat min_concavity_map(image_size, CV_32FC1);
  depth_segmenter_.computeMinConcavityMap(depth_map, concave_normals,
                                          &min_concavity_map);

  // static const std::string kNormalWindowName = "normalsTest";
  // cv::namedWindow(kNormalWindowName, cv::WINDOW_AUTOSIZE);
  // cv::imshow(kNormalWindowName, concave_normals);
  // static const std::string kDepthWindowName = "depthTest";
  // cv::namedWindow(kDepthWindowName, cv::WINDOW_AUTOSIZE);
  // cv::imshow(kDepthWindowName, depth_map);
  // static const std::string kConcavityWindowName = "concavityTest";
  // cv::namedWindow(kConcavityWindowName, cv::WINDOW_AUTOSIZE);
  // cv::imshow(kConcavityWindowName, min_concavity_map);
  // cv::waitKey(2000);

  EXPECT_EQ(cv::countNonZero(expected_convexity != min_concavity_map), 0);
}
}  // namespace depth_segmentation
DEPTH_SEGMENTATION_TESTING_ENTRYPOINT
