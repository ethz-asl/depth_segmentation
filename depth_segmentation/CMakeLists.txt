cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
find_package(glog_catkin REQUIRED)
find_package(gflags_catkin REQUIRED)
find_package(opencv3_catkin REQUIRED)

catkin_simple()

catkin_add_gtest(test_opencv_bug test/test_this_shit.cpp)
target_link_libraries(test_opencv_bug ${catkin_LIBRARIES})

# catkin_add_gtest(test_depth_segmentation test/test_depth_segmentation.cpp)
# target_link_libraries(test_depth_segmentation ${PROJECT_NAME} pthread)


cs_install()
cs_export()
