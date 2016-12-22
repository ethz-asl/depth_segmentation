cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)

if(APPLE)
  catkin_simple()
  set(CMAKE_C_COMPILER clang-omp)
  set(CMAKE_CXX_COMPILER clang-omp++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
  catkin_simple(ALL_DEPS_REQUIRED)
endif()

# OpenMP
 # Adapted from http://wiki.ros.org/catkin/migrating_from_rosbuild
if(OPENMP_FOUND)
  message(STATUS "OpenMP found! cxx_flags: ${OpenMP_CXX_FLAGS} libs: ${OpenMP_LIBS}")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set(OpenMP_LIBS gomp)
  add_definitions(-Wno-unknown-pragmas ${OpenMP_FLAGS})
endif()

cs_add_library(${PROJECT_NAME}
  src/depth_segmentation.cpp
)

# add_definitions(-DWRITE_IMAGES)
cs_add_executable(${PROJECT_NAME}_node
  src/depth_segmentation_node.cpp
)
target_compile_options(${PROJECT_NAME}_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${OpenMP_LIBS})

# COPY TEST DATA
# TODO(ff): We should move the test data to an external repo or a cloud at some point.
# add_custom_target(test_data)
# add_custom_command(TARGET test_data
#                    COMMAND rm -rf test_data
#                    COMMAND mkdir -p test_data
#                    COMMAND cp -r ${CMAKE_SOURCE_DIR}/../test_data/depth_segmentation/*
#                    test_data/ || :)

# UNIT TESTS
catkin_add_gtest(test_depth_segmentation test/test_depth_segmentation.cpp)
target_link_libraries(test_depth_segmentation ${PROJECT_NAME} pthread)
# add_dependencies(test_depth_segmentation test_data)

cs_install()
cs_export()
