#include <iostream>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <depth_segmentation/depth_segmentation.h>

class DepthSegmentationNode {
 public:
  DepthSegmentationNode()
      : node_handle_("~"),
        image_transport_(node_handle_),
        camera_info_ready_(false),
        depth_image_sub_(image_transport_, "/camera/depth/image_raw", 1),
        rgb_image_sub_(image_transport_, "/camera/rgb/image_raw", 1),
        depth_info_sub_(node_handle_, "/camera/depth/camera_info", 1),
        rgb_info_sub_(node_handle_, "/camera/rgb/camera_info", 1),
        image_sync_policy_(ImageSyncPolicy(10), depth_image_sub_,
                           rgb_image_sub_),
        camera_info_sync_policy_(CameraInfoSyncPolicy(10), depth_info_sub_,
                                 rgb_info_sub_) {
    image_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::imageCallback, this, _1, _2));
    camera_info_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::cameraInfoCallback, this, _1, _2));
  }

 private:
  ros::NodeHandle node_handle_;
  image_transport::ImageTransport image_transport_;
  tf::TransformBroadcaster transform_broadcaster_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      ImageSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
      CameraInfoSyncPolicy;

  bool camera_info_ready_;

  depth_segmentation::CameraTracker camera_tracker_;

  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> rgb_info_sub_;

  image_transport::SubscriberFilter depth_image_sub_;
  image_transport::SubscriberFilter rgb_image_sub_;

  message_filters::Synchronizer<ImageSyncPolicy> image_sync_policy_;
  message_filters::Synchronizer<CameraInfoSyncPolicy> camera_info_sync_policy_;

  void publish_tf(cv::Mat cv_transform) {
    tf::Transform transform;

    tf::Vector3 translation_tf(cv_transform.at<double>(0, 3),
                               cv_transform.at<double>(1, 3),
                               cv_transform.at<double>(2, 3));

    tf::Matrix3x3 rotation_tf;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        rotation_tf[j][i] = cv_transform.at<double>(j, i);
      }
    }

    transform.setOrigin(translation_tf);
    transform.setBasis(rotation_tf);

    // TODO(ff): Move outside.
    const static std::string world_frame_name = "world";
    const static std::string camera_frame_name = "camera_link";

    transform_broadcaster_.sendTransform(tf::StampedTransform(
        transform, ros::Time::now(), world_frame_name, camera_frame_name));
  }

  void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
                     const sensor_msgs::Image::ConstPtr& rgb_msg) {
    if (camera_info_ready_) {
      cv_bridge::CvImagePtr cv_depth_image;
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      cv_depth_image->image.convertTo(cv_depth_image->image, CV_32FC1,
                                      1.f / 5000.f);

      cv_bridge::CvImagePtr cv_rgb_image;
      cv_rgb_image =
          cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
      cv::Mat bw_image(cv_rgb_image->image.size(), CV_8UC1);

      cvtColor(cv_rgb_image->image, bw_image, cv::COLOR_RGB2GRAY);
      // cv_rgb_image->image.convertTo(cv_rgb_image->image, CV_8UC1);

      if (!camera_tracker_.getRgbImage().empty() &&
          !camera_tracker_.getDepthImage().empty()) {
#ifdef DISPLAY_DEPTH_IMAGES
        // Place both depth images into one.
        cv::Mat old_depth_image = camera_tracker_.getDepthImage();
        cv::Mat new_depth_image = cv_depth_image->image;
        cv::Size size_old_depth = old_depth_image.size();
        cv::Size size_new_depth = new_depth_image.size();
        cv::Mat combined_depth(size_old_depth.height,
                               size_old_depth.width + size_new_depth.width,
                               CV_32FC1);
        cv::Mat left(combined_depth, cv::Rect(0, 0, size_old_depth.width,
                                              size_old_depth.height));
        old_depth_image.copyTo(left);
        cv::Mat right(combined_depth,
                      cv::Rect(size_old_depth.width, 0, size_new_depth.width,
                               size_new_depth.height));
        new_depth_image.copyTo(right);

        // Adjust the colors, such that the depth images look nicer.
        double min;
        double max;
        cv::minMaxIdx(combined_depth, &min, &max);
        combined_depth -= min;
        cv::Mat adjusted_depth;
        cv::convertScaleAbs(combined_depth, adjusted_depth,
                            255.0 / double(max - min));

        cv::imshow("Depth Images", adjusted_depth);
        cv::waitKey(0);
#endif  // DISPLAY_DEPTH_IMAGES

        // Compute transform from tracker.
        if (camera_tracker_.computeTransform(bw_image, cv_depth_image->image)) {
          // LOG(INFO) << camera_tracker_.getTransform();
          publish_tf(camera_tracker_.getWorldTransform());
          // Update the member images to the new images.
          // TODO(ff): Consider only doing this, when we are far enough away
          // from a frame. (Which basically means we would set a keyframe.)
          camera_tracker_.setRgbImage(bw_image);
          camera_tracker_.setDepthImage(cv_depth_image->image);
        } else {
          LOG(ERROR) << "Failed to compute Transform.";
        }
      } else {
        camera_tracker_.setRgbImage(bw_image);
        camera_tracker_.setDepthImage(cv_depth_image->image);
      }
    }
  }

  void cameraInfoCallback(
      const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg,
      const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg) {
    if (camera_info_ready_) {
      return;
    }

    sensor_msgs::CameraInfo depth_info;
    depth_info = *depth_camera_info_msg;
    Eigen::Vector2d depth_image_size(depth_info.width, depth_info.height);

    cv::Mat K_depth = cv::Mat::eye(3, 3, CV_32FC1);
    K_depth.at<float>(0, 0) = depth_info.K[0];
    K_depth.at<float>(0, 2) = depth_info.K[2];
    K_depth.at<float>(1, 1) = depth_info.K[4];
    K_depth.at<float>(1, 2) = depth_info.K[5];
    K_depth.at<float>(2, 2) = depth_info.K[8];

    sensor_msgs::CameraInfo rgb_info;
    rgb_info = *rgb_camera_info_msg;
    Eigen::Vector2d rgb_image_size(rgb_info.width, rgb_info.height);

    cv::Mat K_rgb = cv::Mat::eye(3, 3, CV_32FC1);
    K_rgb.at<float>(0, 0) = rgb_info.K[0];
    K_rgb.at<float>(0, 2) = rgb_info.K[2];
    K_rgb.at<float>(1, 1) = rgb_info.K[4];
    K_rgb.at<float>(1, 2) = rgb_info.K[5];
    K_rgb.at<float>(2, 2) = rgb_info.K[8];
    // const static std::string kOdometryMethod = "RgbdICPOdometry";
    const static std::string kOdometryMethod = "RgbdOdometry";
    // const static std::string kOdometryMethod = "ICPOdometry";

    camera_tracker_.initialize(depth_image_size.x(), depth_image_size.y(),
                               K_rgb, K_depth, kOdometryMethod);

    camera_info_ready_ = true;
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "depth_segmentation_node");
#ifdef DISPLAY_DEPTH_IMAGES
  cv::namedWindow("Depth Images", cv::WINDOW_AUTOSIZE);
#endif  // DISPLAY_DEPTH_IMAGES
  DepthSegmentationNode depth_segmentation_node;

  while (ros::ok()) {
    ros::spin();
  }

  return EXIT_SUCCESS;
}
