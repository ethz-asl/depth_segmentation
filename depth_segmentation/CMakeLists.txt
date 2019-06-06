cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation)

add_definitions(-std=c++14)

find_package(catkin_simple REQUIRED)

if(APPLE)
  catkin_simple()
  set(LLVM_PATH "/usr/local/opt/llvm")
  find_package(LLVM HINTS ${LLVM_PATH}/lib/cmake/llvm)
  if (LLVM_FOUND)
    message(STATUS "Building with LLVM clang and omp support.")
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_PATH}/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_PATH}/lib")
  else()
    message(STATUS "LLVM not found, building without omp support.")
  endif()
else()
  catkin_simple(ALL_DEPS_REQUIRED)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(OpenMP_FLAGS -fopenmp)
    set(OpenMP_LIBS omp)
    add_definitions(-Wno-unknown-pragmas -fopenmp)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(OpenMP_FLAGS -fopenmp)
    set(OpenMP_LIBS gomp)
    add_definitions(-Wno-unknown-pragmas -fopenmp)
  else()
    message("Unrecognized C++ compiler: ${CMAKE_CXX_COMPILER} ID: ${CMAKE_CXX_COMPILER_ID}. Compiling without openMP support.")
  endif()
endif()

set(WITH_MASKRCNNROS ON)

if (WITH_MASKRCNNROS)
  find_package(mask_rcnn_ros QUIET)
  if (NOT mask_rcnn_ros_FOUND)
    message(WARNING "The semantic instance segmentation mode is enabled but the mask_rcnn_ros package could not be found, trying to build it.")
    execute_process (
        COMMAND catkin build mask_rcnn_ros
        OUTPUT_VARIABLE build_log
        ERROR_VARIABLE build_log
    )
    message(STATUS build_log)
    find_package(mask_rcnn_ros QUIET)
  endif()

  if (mask_rcnn_ros_FOUND)
    message(STATUS "mask_rcnn_ros is found.")
    add_definitions(-DMASKRCNNROS_AVAILABLE)
    include_directories(include ${catkin_INCLUDE_DIRS} ${mask_rcnn_ros_INCLUDE_DIRS} )
  else()
    message(WARNING "Could not find the mask_rcnn_ros package, please add the mask_rcnn_ros package to the current workspace for semantic instance segmentation support.")
  endif()
endif()


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
