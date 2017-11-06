#include <glog/logging.h>
#include <gtest/gtest.h>

#include "depth_segmentation/common.h"
#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/testing_entrypoint.h"

namespace depth_segmentation {

class DepthSegmentationTest : public ::testing::Test {
 protected:
  DepthSegmentationTest()
      : depth_camera_(), params_(), depth_segmenter_(depth_camera_, params_) {
    cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_32FC1);
    camera_matrix.at<float>(0, 0) = 574.0527954101562f;
    camera_matrix.at<float>(0, 2) = 319.5f;
    camera_matrix.at<float>(1, 1) = 574.0527954101562f;
    camera_matrix.at<float>(1, 2) = 239.5f;
    camera_matrix.at<float>(2, 2) = 1.0f;
    depth_camera_.initialize(480u, 640u, CV_32FC1, camera_matrix);
    params_.min_convexity.window_size = 3u;
    depth_segmenter_.initialize();
    params_.normals.method = SurfaceNormalEstimationMethod::kDepthWindowFilter;
    params_.normals.window_size = 3u;
  }
  virtual ~DepthSegmentationTest() {}
  virtual void SetUp() {}

  Params params_;
  DepthCamera depth_camera_;
  DepthSegmenter depth_segmenter_;
};

TEST_F(DepthSegmentationTest, testNormals) {
  static constexpr size_t kNormalImageWidth = 640u;
  static constexpr size_t kNormalImageHeight = 480u;
  cv::Size image_size(kNormalImageWidth, kNormalImageHeight);
  cv::Mat normals(image_size, CV_32FC3);
  cv::Mat expected_normals(image_size, CV_32FC3);
  cv::Mat depth_map(image_size, CV_32FC3);
  cv::Vec3f xz_to_left_normal;
  xz_to_left_normal[0] = -cv::sqrt(2.0) / 2.0f;
  xz_to_left_normal[1] = 0.0f;
  xz_to_left_normal[2] = -cv::sqrt(2.0) / 2.0f;

  // TODO(ff): This is expected only because of the implementation. Consider to
  // check only some range for the normals at boundary conditions.
  cv::Vec3f xz_slight_left_normal;
  xz_slight_left_normal[0] = cos(21.0 / 32.0 * CV_PI);
  xz_slight_left_normal[1] = 0.0f;
  xz_slight_left_normal[2] = cos(27.0 / 32.0 * CV_PI);

  cv::Vec3f z_normal;
  z_normal[0] = 0.0f;
  z_normal[1] = 0.0f;
  z_normal[2] = -1.0f;

  const float fx = depth_camera_.getCameraMatrix().at<float>(0, 0);
  const float fy = depth_camera_.getCameraMatrix().at<float>(1, 1);
  const float cx = depth_camera_.getCameraMatrix().at<float>(0, 2);
  const float cy = depth_camera_.getCameraMatrix().at<float>(1, 2);

  const float kZMinDistance = 2.0f;
  const float kZStep = 1.0f / fx;

  float z_distance = kZMinDistance;

  for (size_t y = 0u; y < kNormalImageHeight; ++y) {
    z_distance = kZMinDistance;
    for (size_t x = 0u; x < kNormalImageWidth; ++x) {
      depth_map.at<cv::Vec3f>(y, x) =
          cv::Vec3f((x - cx) / fx, (y - cy) / fy, z_distance);
      if (x < kNormalImageWidth / 2u - 1u) {
        expected_normals.at<cv::Vec3f>(y, x) = z_normal;
      } else if (x < kNormalImageWidth / 2u) {
        expected_normals.at<cv::Vec3f>(y, x) = xz_slight_left_normal;
        z_distance -= kZStep;
      } else {
        expected_normals.at<cv::Vec3f>(y, x) = xz_to_left_normal;
        z_distance -= kZStep;
      }
    }
  }

  depth_segmenter_.computeNormalMap(depth_map, &normals);

  cv::viz::Viz3d viz_3d("Pointcloud with Normals");
  visualizeDepthMapWithNormals(depth_map, normals, &viz_3d);

  for (size_t y = 0u; y < kNormalImageHeight; ++y) {
    for (size_t x = 0u; x < kNormalImageWidth; ++x) {
      cv::Vec3f normal = normals.at<cv::Vec3f>(y, x);
      cv::Vec3f expected_normal = expected_normals.at<cv::Vec3f>(y, x);
      if (x < kNormalImageWidth / 2u) {
        for (size_t k = 0u; k < normals.channels(); ++k) {
          // Be generous with the normal error at plane intersection.
          EXPECT_NEAR(normal(k), expected_normal(k), 1.0e-3);
        }
      } else {
        for (size_t k = 0u; k < normals.channels(); ++k) {
          EXPECT_NEAR(normal(k), expected_normal(k), 1.0e-5);
        }
      }
    }
  }
}

TEST_F(DepthSegmentationTest, testNormals2) {
  static constexpr size_t kNormalImageWidth = 640u;
  static constexpr size_t kNormalImageHeight = 480u;
  cv::Size image_size(kNormalImageWidth, kNormalImageHeight);
  cv::Mat normals(image_size, CV_32FC3);
  cv::Mat expected_normals(image_size, CV_32FC3);
  cv::Mat depth_map(image_size, CV_32FC3);
  cv::Vec3f yz_up_normal;
  yz_up_normal[0] = 0.0f;
  yz_up_normal[1] = -cv::sqrt(2.0) / 2.0f;
  yz_up_normal[2] = -cv::sqrt(2.0) / 2.0f;
  cv::Vec3f z_normal;
  z_normal[0] = 0.0f;
  z_normal[1] = 0.0f;
  z_normal[2] = -1.0f;

  const float fx = depth_camera_.getCameraMatrix().at<float>(0, 0);
  const float fy = depth_camera_.getCameraMatrix().at<float>(1, 1);
  const float cx = depth_camera_.getCameraMatrix().at<float>(0, 2);
  const float cy = depth_camera_.getCameraMatrix().at<float>(1, 2);

  const float kZMinDistance = 2.0f;
  const float kZStep = 1.0f / fy;

  float z_distance = kZMinDistance;

  for (size_t x = 0u; x < kNormalImageWidth; ++x) {
    z_distance = kZMinDistance;
    for (size_t y = 0u; y < kNormalImageHeight; ++y) {
      depth_map.at<cv::Vec3f>(y, x) =
          cv::Vec3f((x - cx) / fx, (y - cy) / fy, z_distance);
      if (y < kNormalImageHeight / 2u - 1u) {
        expected_normals.at<cv::Vec3f>(y, x) = z_normal;
      } else {
        expected_normals.at<cv::Vec3f>(y, x) = yz_up_normal;
        z_distance -= kZStep;
      }
    }
  }
  depth_segmenter_.computeNormalMap(depth_map, &normals);

  cv::viz::Viz3d viz_3d("Pointcloud with Normals");
  visualizeDepthMapWithNormals(depth_map, normals, &viz_3d);

  EXPECT_EQ(cv::countNonZero(normals != expected_normals), 0);
}

TEST_F(DepthSegmentationTest, testConvexity) {
  static constexpr size_t kNormalImageWidth = 640u;
  static constexpr size_t kNormalImageHeight = 480u;
  cv::Size image_size(kNormalImageWidth, kNormalImageHeight);
  cv::Mat concave_normals(image_size, CV_32FC3);
  cv::Mat depth_map(image_size, CV_32FC3);
  cv::Vec3f xz_to_right_normal;
  xz_to_right_normal[0] = cv::sqrt(2.0) / 2.0f;
  xz_to_right_normal[1] = 0.0f;
  xz_to_right_normal[2] = -cv::sqrt(2.0) / 2.0f;
  cv::Vec3f xz_to_left_normal;
  xz_to_left_normal[0] = -cv::sqrt(2.0) / 2.0f;
  xz_to_left_normal[1] = 0.0f;
  xz_to_left_normal[2] = -cv::sqrt(2.0) / 2.0f;
  cv::Vec3f z_normal;
  z_normal[0] = 0.0f;
  z_normal[1] = 0.0f;
  z_normal[2] = -1.0f;

  // We are expecting the concavities at half the image width (2px wide).
  cv::Mat expected_convexity = cv::Mat::ones(image_size, CV_32FC1);
  const float fx = depth_camera_.getCameraMatrix().at<float>(0, 0);
  const float fy = depth_camera_.getCameraMatrix().at<float>(1, 1);
  const float cx = depth_camera_.getCameraMatrix().at<float>(0, 2);
  const float cy = depth_camera_.getCameraMatrix().at<float>(1, 2);

  const float kZMinDistance = 0.2f;
  const float kZStep = 1.0f / fx;

  float z_distance = kZMinDistance;

  for (size_t y = 0u; y < kNormalImageHeight; ++y) {
    z_distance = kZMinDistance;
    for (size_t x = 0u; x < kNormalImageWidth; ++x) {
      depth_map.at<cv::Vec3f>(y, x) =
          cv::Vec3f((x - cx) / fx, (y - cy) / fy, z_distance);
      if (x < kNormalImageWidth / 2u - 1u) {
        concave_normals.at<cv::Vec3f>(y, x) = xz_to_right_normal;
        z_distance += kZStep;
      } else if (x == (kNormalImageWidth / 2u - 1u)) {
        expected_convexity.at<float>(y, x) = 0.0f;
        concave_normals.at<cv::Vec3f>(y, x) = xz_to_right_normal;
        if (y == 0u) {
        }
      } else if (x == (kNormalImageWidth / 2u)) {
        if (y == 0u) {
        }
        expected_convexity.at<float>(y, x) = 0.0f;
        concave_normals.at<cv::Vec3f>(y, x) = xz_to_left_normal;
        z_distance -= kZStep;
      } else {
        concave_normals.at<cv::Vec3f>(y, x) = xz_to_left_normal;
        z_distance -= kZStep;
      }
    }
  }
  cv::Mat min_convexity_map(image_size, CV_32FC1);
  cv::Mat min_convexity_map(image_size, CV_32FC1);
  depth_segmenter_.computeMinConvexityMap(depth_map, concave_normals,
                                          &min_convexity_map,
                                          &min_convexity_map_thresholded);
  static const std::string kDepthWindowName = "depthTest";
  cv::namedWindow(kDepthWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(kDepthWindowName, depth_map);
  static const std::string kConvexityWindowName = "convexityTest";
  cv::namedWindow(kConvexityWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(kConvexityWindowName, min_convexity_map);
  static const std::string kConvexityGTWindowName = "convexityGTTest";
  cv::namedWindow(kConvexityGTWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(kConvexityGTWindowName, expected_convexity);
  cv::viz::Viz3d viz_3d("Pointcloud with Normals");
  visualizeDepthMapWithNormals(depth_map, concave_normals, &viz_3d);

  EXPECT_EQ(cv::countNonZero(expected_convexity != min_convexity_map), 0);
}
}  // namespace depth_segmentation
DEPTH_SEGMENTATION_TESTING_ENTRYPOINT
