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
set(OpenMP_FLAGS -fopenmp)
if(${CMAKE_CXX_COMPILER} STREQUAL clang++)
  set(OpenMP_LIBS omp)
elseif(${CMAKE_CXX_COMPILER} STREQUAL g++)
  set(OpenMP_LIBS gomp)
else()
  message("Unrecognized C++ compiler: ${CMAKE_CXX_COMPILER}")
endif()
  add_definitions(-Wno-unknown-pragmas -fopenmp)

cs_add_library(${PROJECT_NAME}
  src/depth_segmentation.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenMP_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_FLAGS})


# add_definitions(-DWRITE_IMAGES)
cs_add_executable(${PROJECT_NAME}_node
  src/depth_segmentation_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

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

cs_install()
cs_export()
