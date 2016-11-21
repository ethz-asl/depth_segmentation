#ifndef DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
#define DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>

#include "depth_segmentation/common.h"

namespace depth_segmentation {

class Camera {
 public:
  Camera() {}
  Camera(const size_t height, const size_t width, const int type,
         const cv::Mat& camera_matrix)
      : type_(type),
        height_(height),
        width_(width),
        camera_matrix_(camera_matrix) {}
  inline void initialize(const size_t height, const size_t width,
                         const int type, const cv::Mat& camera_matrix) {
    type_ = type;
    height_ = height;
    width_ = width;
    camera_matrix_ = camera_matrix;
  }
  inline void setCameraMatrix(const cv::Mat& camera_matrix) {
    CHECK(!camera_matrix.empty());
    camera_matrix_ = camera_matrix;
  }
  virtual void setImage(const cv::Mat& image) = 0;
  inline void setMask(const cv::Mat& mask) { mask_ = mask; }
  inline void setType(const int type) {
    CHECK(type == CV_8UC1 || type == CV_32FC1);
    type_ = type;
  }
  inline cv::Mat getCameraMatrix() const { return camera_matrix_; }
  inline cv::Mat getImage() const { return image_; }
  inline cv::Mat getMask() const { return mask_; }
  inline int getType() { return type_; }
  inline size_t getHeight() const { return height_; }
  inline size_t getWidth() const { return width_; }

 protected:
  cv::Mat image_;

 private:
  int type_;
  size_t height_;
  size_t width_;
  cv::Mat camera_matrix_;
  cv::Mat mask_;
};

class DepthCamera : public Camera {
 public:
  DepthCamera() {}
  void setImage(const cv::Mat& image) {
    CHECK(!image.empty());
    CHECK(image.type() == CV_32FC1);
    image_ = image;
  }
};

class RgbCamera : public Camera {
 public:
  RgbCamera() {}
  void setImage(const cv::Mat& image) {
    CHECK(!image.empty());
    CHECK(image.type() == CV_8UC1);
    image_ = image;
  }
};

class CameraTracker {
 public:
  CameraTracker();
  void initialize(DepthCamera& depth_camera, RgbCamera& rgb_camera,
                  const std::string odometry_type) {
    setDepthCamera(depth_camera);
    setRgbCamera(rgb_camera);
    return initialize(depth_camera.getWidth(), depth_camera.getHeight(),
                      rgb_camera.getCameraMatrix(),
                      depth_camera.getCameraMatrix(), odometry_type);
  }
  void initialize(const size_t width, const size_t height,
                  const cv::Mat& rgb_camera_matrix,
                  const cv::Mat& depth_camera_matrix,
                  const std::string odometry_type);

  void initialize(const size_t width, const size_t height,
                  const cv::Mat& rgb_camera_matrix,
                  const cv::Mat& depth_camera_matrix) {
    return initialize(width, height, rgb_camera_matrix, depth_camera_matrix,
                      kCameraTrackerNames[CameraTrackerType::kRgbdICPOdometry]);
  }

  bool computeTransform(const cv::Mat& src_rgb_image,
                        const cv::Mat& src_depth_image,
                        const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image,
                        const cv::Mat& src_depth_mask,
                        const cv::Mat& dst_depth_mask);
  bool computeTransform(const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image,
                        const cv::Mat& dst_depth_mask) {
    return computeTransform(getRgbImage(), getDepthImage(), dst_rgb_image,
                            dst_depth_image, getDepthMask(), dst_depth_mask);
  }
  void createMask(const cv::Mat& depth, cv::Mat* mask);
  void dilateFrame(cv::Mat& image, cv::Mat& depth);

  inline void setDepthImage(const cv::Mat& depth_image) {
    depth_camera_->setImage(depth_image);
  }
  inline void setDepthMask(const cv::Mat& depth_mask) {
    depth_camera_->setMask(depth_mask);
  }
  inline void setRgbImage(const cv::Mat& rgb_image) {
    rgb_camera_->setImage(rgb_image);
  }
  inline void setDepthCameraMatrix(const cv::Mat& camera_matrix) {
    depth_camera_->setCameraMatrix(camera_matrix);
  }
  inline void setRgbCameraMatrix(const cv::Mat& camera_matrix) {
    rgb_camera_->setCameraMatrix(camera_matrix);
  }
  inline void setDepthCamera(DepthCamera& depth_camera) {
    depth_camera_ = &depth_camera;
  }
  inline void setRgbCamera(RgbCamera& rgb_camera) { rgb_camera_ = &rgb_camera; }
  inline cv::Mat getTransform() const { return transform_; }
  inline cv::Mat getWorldTransform() const { return world_transform_; }
  inline cv::Mat getRgbImage() const { return rgb_camera_->getImage(); }
  inline cv::Mat getDepthImage() const { return depth_camera_->getImage(); }
  inline cv::Mat getDepthMask() const { return depth_camera_->getMask(); }
  inline DepthCamera* getDepthCamera() const { return depth_camera_; }
  inline RgbCamera* getRgbCamera() const { return rgb_camera_; }

  void visualize(const cv::Mat old_depth_image,
                 const cv::Mat new_depth_image) const;

  enum CameraTrackerType {
    kRgbdICPOdometry = 0,
    kRgbdOdometry = 1,
    kICPOdometry = 2
  };
  static constexpr size_t kImageRange = 255;
  static constexpr double kMinDepth = 0.15;
  static constexpr double kMaxDepth = 10.0;
  const std::vector<std::string> kCameraTrackerNames = {
      "RgbdICPOdometry", "RgbdOdometry", "ICPOdometry"};

 private:
  DepthCamera* depth_camera_;
  RgbCamera* rgb_camera_;
  cv::Ptr<cv::rgbd::Odometry> odometry_;
  cv::Mat transform_;
  cv::Mat world_transform_;
};

class DepthSegmenter {
 public:
  DepthSegmenter(){};
  void initialize(DepthCamera& depth_camera);
  void computeDepthMap(const cv::Mat& depth_image, cv::Mat* depth_map);
  void computeMaxDistanceMap(const cv::Mat& image, cv::Mat* distance_map);
  void computeNormalMap(const cv::Mat& depth_map, cv::Mat* normal_map);
  void computeMinConcavityMap(const cv::Mat& image, cv::Mat* concavity_map);
  void convertToConcavityAwareNormalMap(const cv::Mat& concavity_map,
                                        const cv::Mat& normal_map,
                                        cv::Mat* combined_map);
  void edgeMap(const cv::Mat& image, cv::Mat* edge_map);
  void labelMap(const cv::Mat& edge_map, cv::Mat* labeled_map);
  inline void setDepthCamera(DepthCamera& depth_camera) {
    depth_camera_ = &depth_camera;
  }
  inline void setDepthCameraMatrix(const cv::Mat& camera_matrix) {
    depth_camera_->setCameraMatrix(camera_matrix);
  }
  inline DepthCamera* getDepthCamera() const { return depth_camera_; }

 private:
  DepthCamera* depth_camera_;

  cv::rgbd::RgbdNormals rgbd_normals_;
};
}

#endif  // DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
