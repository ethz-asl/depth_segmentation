cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/depth_segmentation.cpp
)

# add_definitions(-DWRITE_IMAGES)
# add_definitions(-DDISPLAY_DEPTH_IMAGES)
cs_add_executable(${PROJECT_NAME}_node
  src/depth_segmentation_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

cs_install()
cs_export()
