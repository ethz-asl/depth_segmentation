/********************************************************************
**                Image Component Library (ICL)                    **
**                                                                 **
** Copyright (C) 2006-2013 CITEC, University of Bielefeld          **
**                         Neuroinformatics Group                  **
** Website: www.iclcv.org and                                      **
**          http://opensource.cit-ec.de/projects/icl               **
**                                                                 **
** File   : depth_segmentation_node.cpp                            **
** Module : ICLGeom                                                **
** Authors: Andre Ueckermann (original file in icl)                **
**          Fadri Furrer <fadri.furrer@mavt.ethz.ch> (ROS node)    **
**                                                                 **
** GNU LESSER GENERAL PUBLIC LICENSE                               **
** This file may be used under the terms of the GNU Lesser General **
** Public License version 3.0 as published by the                  **
**                                                                 **
** Free Software Foundation and appearing in the file LICENSE.LGPL **
** included in the packaging of this file.  Please review the      **
** following information to ensure the license requirements will   **
** be met: http://www.gnu.org/licenses/lgpl-3.0.txt                **
**                                                                 **
** The development of this software was supported by the           **
** Excellence Cluster EXC 277 Cognitive Interaction Technology.    **
** The Excellence Cluster EXC 277 is a grant of the Deutsche       **
** Forschungsgemeinschaft (DFG) in the context of the German       **
** Excellence Initiative.                                          **
**                                                                 **
********************************************************************/

// This has to be defined, otherwise we are getting namespace
// conflicts, as `cv` is defined in opencv and also in the ICL library.
#define ICL_NO_USING_NAMESPACES
#include <cv.h>

#include <ICLCore/CCFunctions.h>
#include <ICLCore/CoreFunctions.h>
#include <ICLCore/OpenCV.h>
#include <ICLCore/PseudoColorConverter.h>
#include <ICLGeom/ConfigurableDepthImageSegmenter.h>
#include <ICLGeom/Scene.h>
#include <ICLQt/Common.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/pca.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/pcl_config.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int64.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZRGB PointType;

using namespace icl::core;
using namespace icl::geom;
using namespace icl::utils;
using namespace icl::qt;

icl::qt::HSplit gui;

ros::Publisher point_cloud2_pub;
ros::Publisher segment_size_pub;

// GenericGrabber grabDepth, grabColor;
int KINECT_CAM = 0, VIEW_CAM = 1;

Camera rgb_cam;
Camera depth_cam;

// ImgBase* depth_image;
ImgParams rgb_params;
ImgParams depth_params;
ImgBase* rgb_image;
ImgBase* depth_image;

uint64 rgb_width;
uint64 rgb_height;
uint64 depth_width;
uint64 depth_height;

constexpr uint64 kMinSegmentSize = 1000;
constexpr uint64 kMinLabeledPointSize = 100;
constexpr uint64 kColorConverterMaxValue = 4000;
constexpr double kMinZDistance = 0.2;
constexpr double kMaxZDistance = 2.0;
constexpr double kThresholdDist = 10000;
constexpr double kPcaMaxEigenValueScale = 1e-2;
constexpr bool kPointCloudIsOrganized = true;
constexpr bool kPointCloudWithNormals = true;
constexpr bool kPointCloudWithColors = true;
constexpr bool kPointCloudWithLabels = true;

bool write_segment_files = false;
bool filter_thin_elements = true;
bool rgb_info_ready = false;
bool depth_info_ready = false;
bool rgb_ready = false;
bool depth_ready = true;
bool initialized = false;

// DepthPreference depth_preference;
cv_bridge::CvImagePtr cv_depth_image;
cv_bridge::CvImagePtr cv_rgb_image;

PseudoColorConverter* pseudo_color_converter;
ConfigurableDepthImageSegmenter* segmentation;
PointCloudObject* point_cloud_object;
Scene scene;

struct AdaptedSceneMouseHandler : public MouseHandler {
  Mutex mutex;
  MouseHandler* h;

  AdaptedSceneMouseHandler(MouseHandler* h) : h(h) {}

  void process(const MouseEvent& e) {
    Mutex::Locker l(mutex);
    h->process(e);
  }

}* mouse = 0;

void rgbCameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  if (rgb_info_ready) {
    return;
  } else {
    rgb_width = msg->width;
    rgb_height = msg->height;
    rgb_params = ImgParams(rgb_width, rgb_height, formatRGB);
    rgb_cam.setPrincipalPointOffset(msg->K[2], msg->K[5]);
    rgb_cam.setFocalLength(msg->K[0]);
    rgb_image = imgNew(depth8u, rgb_params);
    rgb_cam = Camera();
    rgb_cam.setResolution(Size(rgb_width, rgb_height));
    rgb_info_ready = true;
    ROS_INFO("Got rgb camera info.");
  }
}

void depthCameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  if (depth_info_ready) {
    return;
  } else {
    depth_width = msg->width;
    depth_height = msg->height;
    depth_params = ImgParams(depth_width, depth_height, formatMatrix);
    depth_cam.setPrincipalPointOffset(msg->K[2], msg->K[5]);
    depth_cam.setFocalLength(msg->K[0]);
    depth_image = imgNew(depth32f, depth_params);
    depth_cam = Camera();
    depth_cam.setResolution(Size(depth_width, depth_height));
    depth_info_ready = true;
    ROS_INFO("Got depth camera info.");
  }
}

void rgbImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  if (!rgb_info_ready || !initialized) {
    return;
  }
  // TODO(ff): Add a mutex here.
  rgb_ready = false;
  cv_rgb_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  cv::Mat rgb_mat;
  cv_rgb_image->image.convertTo(rgb_mat, CV_BGR2RGB);
  mat_to_img(&rgb_mat, rgb_image);
  rgb_ready = true;
}

void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  if (!depth_info_ready || !initialized) {
    return;
  }
  // TODO(ff): Add a mutex here.
  depth_ready = false;
  cv_depth_image =
      cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);

  cv::Mat float_depth_map;
  cv_depth_image->image.convertTo(float_depth_map, CV_32FC1, 1, 0.0f);

  cv::Mat truncated_depth_map;
  cv::threshold(float_depth_map, truncated_depth_map, kThresholdDist, 1,
                cv::THRESH_TRUNC);

  cv::Mat no_zero_values = truncated_depth_map.clone();
  no_zero_values.setTo(0.0f, (truncated_depth_map == kThresholdDist));

  mat_to_img(&no_zero_values, depth_image);
  depth_ready = true;
}

void init() {
  while (!rgb_info_ready || !depth_info_ready) {
    ROS_INFO("Spinning, waiting for rgb and depth camera info messages.");
    ros::spinOnce();
    ros::Duration(1.0).sleep();
  }

  Size size = pa("-size");

  if (pa("-d")) {  // get depth cam
    std::string depth_cam_name = pa("-d").as<std::string>();
    depth_cam = Camera(depth_cam_name);
    depth_cam.setResolution(size);
  } else {  // default depth cam
    depth_cam = Camera();
    depth_cam.setResolution(size);
    depth_cam.setName("Depth Camera");
  }

  // initialize depth heatmap mapping
  std::vector<PseudoColorConverter::Stop>
      color_converter_stops;  // create heatmap
  Color red(255, 0, 0);
  Color yellow(255, 255, 0);
  Color green(0, 255, 0);
  Color cyan(0, 255, 255);
  Color blue(0, 0, 255);
  // TODO(ff): Make the ColorConverter stops params.
  color_converter_stops.push_back(PseudoColorConverter::Stop(0.25, red));
  color_converter_stops.push_back(PseudoColorConverter::Stop(0.35, yellow));
  color_converter_stops.push_back(PseudoColorConverter::Stop(0.45, green));
  color_converter_stops.push_back(PseudoColorConverter::Stop(0.55, cyan));
  color_converter_stops.push_back(PseudoColorConverter::Stop(0.8, blue));
  pseudo_color_converter =
      new PseudoColorConverter(color_converter_stops, kColorConverterMaxValue);

  point_cloud_object = new PointCloudObject(
      size.width, size.height, kPointCloudIsOrganized, kPointCloudWithNormals,
      kPointCloudWithColors, kPointCloudWithLabels);

  if (pa("-fcpu")) {
    segmentation = new ConfigurableDepthImageSegmenter(
        ConfigurableDepthImageSegmenter::CPU, depth_cam,
        PointCloudCreator::DepthImageMode::DistanceToCamPlane);
  } else if (pa("-fgpu")) {
    segmentation = new ConfigurableDepthImageSegmenter(
        ConfigurableDepthImageSegmenter::GPU, depth_cam,
        PointCloudCreator::DepthImageMode::DistanceToCamPlane);
  } else {
    segmentation = new ConfigurableDepthImageSegmenter(
        ConfigurableDepthImageSegmenter::BEST, depth_cam,
        PointCloudCreator::DepthImageMode::DistanceToCamPlane);
  }
  // TODO(ff): Make these ROS params and generate a dynamic reconfigure.
  segmentation->setPropertyValue("general.enable segmentation", true);
  segmentation->setPropertyValue("general.stabelize segmentation", true);
  segmentation->setPropertyValue("general.depth scaling", 1.0);
  segmentation->setPropertyValue("general.use ROI", false);
  segmentation->setPropertyValue("general.ROI min x", -1200);
  segmentation->setPropertyValue("general.ROI max x", 1200);
  segmentation->setPropertyValue("general.ROI min y", -100);
  segmentation->setPropertyValue("general.ROI max y", 1050);
  segmentation->setPropertyValue("general.ROI min z", 0);
  segmentation->setPropertyValue("general.ROI max z", 1050);

  segmentation->setPropertyValue("pre.enable temporal smoothing", true);
  segmentation->setPropertyValue("pre.temporal smoothing size", 6);
  segmentation->setPropertyValue("pre.temporal smoothing diff", 10);
  segmentation->setPropertyValue("pre.filter", "median5x5");
  segmentation->setPropertyValue("pre.normal range", 1);
  segmentation->setPropertyValue("pre.averaging", true);
  segmentation->setPropertyValue("pre.averaging range", 2);
  segmentation->setPropertyValue("pre.smoothing", "linear");
  segmentation->setPropertyValue("pre.edge threshold", 0.9);
  segmentation->setPropertyValue("pre.edge angle method", "mean");
  segmentation->setPropertyValue("pre.edge neighborhood", 1.2);

  segmentation->setPropertyValue("surfaces.min surface size", 45);
  segmentation->setPropertyValue("surfaces.assignment radius", 7);
  segmentation->setPropertyValue("surfaces.assignment distance", 10.0);

  segmentation->setPropertyValue("cutfree.enable cutfree adjacency feature",
                                 true);
  segmentation->setPropertyValue("cutfree.ransac euclidean distance", 8.0);
  segmentation->setPropertyValue("cutfree.ransac passes", 20);
  segmentation->setPropertyValue("cutfree.ransac tolerance", 30);
  segmentation->setPropertyValue("cutfree.min angle", 30.0);

  segmentation->setPropertyValue("coplanarity.enable coplanarity feature",
                                 true);
  segmentation->setPropertyValue("coplanarity.max angle", 30.0);
  segmentation->setPropertyValue("coplanarity.distance tolerance", 1.5);
  segmentation->setPropertyValue("coplanarity.outlier tolerance", 5.0);
  segmentation->setPropertyValue("coplanarity.num triangles", 20);
  segmentation->setPropertyValue("coplanarity.num scanlines", 9);

  segmentation->setPropertyValue("curvature.enable curvature feature", true);
  segmentation->setPropertyValue("curvature.histogram similarity", 0.5);
  segmentation->setPropertyValue("curvature.enable open objects", true);
  segmentation->setPropertyValue("curvature.max distance", 10);
  segmentation->setPropertyValue("curvature.enable occluded objects", true);
  segmentation->setPropertyValue("curvature.max error", 10.0);
  segmentation->setPropertyValue("curvature.ransac passes", 20);
  segmentation->setPropertyValue("curvature.distance tolerance", 3.0);
  segmentation->setPropertyValue("curvature.outlier tolerance", 5.0);

  segmentation->setPropertyValue("remaining.enable remaining points feature",
                                 true);
  segmentation->setPropertyValue("remaining.min size", 10);
  segmentation->setPropertyValue("remaining.euclidean distance", 10.0);
  segmentation->setPropertyValue("remaining.radius", 0);
  segmentation->setPropertyValue("remaining.assign euclidean distance", 10.0);
  segmentation->setPropertyValue("remaining.support tolerance", 9);

  segmentation->setPropertyValue("graphcut.threshold", 0.5);

  GUI controls = HBox().minSize(12, 12);

  controls << (VBox() << Button("reset view").handle("resetView")
                      << Fps(10).handle("fps")
                      << Prop("segmentation").minSize(10, 8));

  gui << (VBox() << Draw3D().handle("hdepth").minSize(10, 8)
                 << Draw3D().handle("hcolor").minSize(10, 8))
      << (VBox() << Draw3D().handle("hedge").minSize(10, 8)
                 << Draw3D().handle("hnormal").minSize(10, 8))
      << (HSplit() << Draw3D().handle("draw3D").minSize(40, 30) << controls)
      << Show();

  // Depth camera
  scene.addCamera(depth_cam);

  // View camera
  scene.addCamera(depth_cam);

  if (pa("-d")) {
    scene.setDrawCoordinateFrameEnabled(true);
  } else {
    scene.setDrawCoordinateFrameEnabled(false);
  }

  scene.setDrawCamerasEnabled(true);

  scene.addObject(point_cloud_object);
  scene.setBounds(1000);

  DrawHandle3D draw_handle_3d = gui["draw3D"];

  mouse = new AdaptedSceneMouseHandler(scene.getMouseHandler(VIEW_CAM));
  draw_handle_3d->install(mouse);

  scene.setLightingEnabled(false);
  point_cloud_object->setPointSize(3);

  initialized = true;
  ROS_INFO("initialized");
}

void run() {
  if (!depth_ready || !rgb_ready || !initialized) {
    ros::spinOnce();
    return;
  }
  depth_ready = false;
  rgb_ready = false;
  ButtonHandle resetView = gui["resetView"];
  DrawHandle3D hdepth = gui["hdepth"];
  DrawHandle3D hcolor = gui["hcolor"];
  DrawHandle3D hedge = gui["hedge"];
  DrawHandle3D hnormal = gui["hnormal"];

  // reset camera view
  if (resetView.wasTriggered()) {
    scene.getCamera(VIEW_CAM) = scene.getCamera(KINECT_CAM);
  }

  // create heatmap
  static ImgBase* heatmap_image = 0;
  pseudo_color_converter->apply(depth_image, &heatmap_image);

  // segment
  segmentation->apply(*depth_image->as32f(), *point_cloud_object);

  std::vector<std::vector<int>> surfaces = segmentation->getSurfaces();
  std::vector<std::vector<int>> segmentation_segments =
      segmentation->getSegments();
  std::vector<std::vector<int>> segment_points;
  for (std::vector<int> surface_indices : segmentation_segments) {
    std::vector<int> point_ids;
    for (int surface_index : surface_indices) {
      for (int point_id : surfaces[surface_index]) {
        point_ids.push_back(point_id);
      }
    }
    segment_points.push_back(point_ids);
  }
  ros::Time ros_now = ros::Time::now();
  std::vector<sensor_msgs::PointCloud2> segments;
  uint64_t index = 0;
  for (std::vector<int> labeled_points : segment_points) {
    if (labeled_points.size() < kMinLabeledPointSize) {
      break;
    }
    bool forget_segment = false;
    pcl::PointCloud<PointType>::Ptr segmented_point_cloud_pcl(
        new pcl::PointCloud<PointType>);
    for (size_t point_id : labeled_points) {
      icl::math::FixedColVector<float, 3> xyz =
          point_cloud_object->selectXYZ()[point_id];
      icl::math::FixedColVector<float, 4> rgba =
          point_cloud_object->selectRGBA32f()[point_id];
      PointType point_pcl;
      point_pcl.x = xyz[1] / 1000.0f;
      point_pcl.y = xyz[0] / 1000.0f;
      point_pcl.z = -xyz[2] / 1000.0f;

      point_pcl.r = (int)(rgba[0] * 256);
      point_pcl.g = (int)(rgba[1] * 256);
      point_pcl.b = (int)(rgba[2] * 256);
      // TODO(ff): Improve this, for now, just skipping segments that contain
      // points too close to the camera and too far away.
      if (std::abs(point_pcl.z) <= kMinZDistance ||
          std::abs(point_pcl.z) > kMaxZDistance) {
        continue;
      }
      segmented_point_cloud_pcl->push_back(point_pcl);
    }
    // Throw away segments that have a too small size.
    if (segmented_point_cloud_pcl->size() <= kMinSegmentSize) {
      continue;
    }
    // Throw away segments that are too flat or thin, as they will merge with
    // all flat/thin segments.
    pcl::PCA<PointType> pca(*segmented_point_cloud_pcl);
    Eigen::VectorXf eigen_values = pca.getEigenValues();
    if (filter_thin_elements &&
        eigen_values.minCoeff() <
            kPcaMaxEigenValueScale * eigen_values.maxCoeff()) {
      continue;
    }

    if (write_segment_files) {
      pcl::io::savePLYFileASCII(std::to_string(ros_now.toNSec()) + "_segment_" +
                                    std::to_string(index) + ".ply",
                                *segmented_point_cloud_pcl);
    }
    sensor_msgs::PointCloud2 pcl2_msg;
    pcl::toROSMsg(*segmented_point_cloud_pcl, pcl2_msg);
    pcl2_msg.header.stamp = ros_now;
    pcl2_msg.header.frame_id = "/camera_depth_optical_frame";
    segments.push_back(pcl2_msg);
    ++index;
  }
  std_msgs::Int64 segment_size_msg;
  segment_size_msg.data = segments.size();
  segment_size_pub.publish(segment_size_msg);

  for (sensor_msgs::PointCloud2 segment : segments) {
    point_cloud2_pub.publish(segment);
  }

  Img8u edge_image = segmentation->getEdgeImage();
  Img8u normal_image = segmentation->getNormalImage();

  // display
  hdepth = heatmap_image;
  hcolor = rgb_image;
  hedge = &edge_image;
  hnormal = &normal_image;

  gui["fps"].render();
  hdepth.render();
  hcolor.render();
  hedge.render();
  hnormal.render();

  gui["draw3D"].link(scene.getGLCallback(VIEW_CAM));
  gui["draw3D"].render();
  ros::spinOnce();
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "depth_segmentation_node");
  ros::NodeHandle node_handle("~");
  ros::Subscriber rgb_info_sub;
  ros::Subscriber depth_info_sub;
  ros::Subscriber rgb_image_sub;
  ros::Subscriber depth_image_sub;

  std::string rgb_camera_info_topic;
  std::string depth_camera_info_topic;
  std::string rgb_image_topic;
  std::string depth_image_topic;

  node_handle.param<std::string>("rgb_camera_info_topic", rgb_camera_info_topic,
                                 "/camera/rgb/camera_info");
  node_handle.param<std::string>("depth_camera_info_topic",
                                 depth_camera_info_topic,
                                 "/camera/depth/camera_info");
  node_handle.param<std::string>("rgb_image_topic", rgb_image_topic,
                                 "/camera/rgb/image_raw");
  node_handle.param<std::string>("depth_image_topic", depth_image_topic,
                                 "/camera/depth/image_raw");

  depth_info_sub = node_handle.subscribe(depth_camera_info_topic, 1,
                                         depthCameraInfoCallback);
  rgb_info_sub =
      node_handle.subscribe(rgb_camera_info_topic, 1, rgbCameraInfoCallback);

  depth_image_sub =
      node_handle.subscribe(depth_image_topic, 1, depthImageCallback);
  rgb_image_sub = node_handle.subscribe(rgb_image_topic, 1, rgbImageCallback);
  point_cloud2_pub =
      node_handle.advertise<sensor_msgs::PointCloud2>("object_segment", 1000);
  segment_size_pub =
      node_handle.advertise<std_msgs::Int64>("segment_size", 1000);

  return ICLApp(argc, argv,
                "-size|-s(Size=VGA) -depth-cam|-d(file) -fcpu|force-cpu "
                "-fgpu|force-gpu)",
                init, run)
      .exec();
}
