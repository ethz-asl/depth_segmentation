cmake_minimum_required(VERSION 2.8.3)
project(iclcv_segmentation)

add_definitions(-std=c++14)
add_definitions(-DICL_HAVE_OPENGL)

find_package(catkin_simple REQUIRED)

find_package(Qt5Core)

SET(ICL_QT_MODULES Core Gui Multimedia Widgets OpenGL)

SET(ICL_ALL_QT_LIBS_FOUND TRUE)
SET(QT_ROOT /usr/local/opt/qt5)
FOREACH(M ${ICL_QT_MODULES})
  FIND_PACKAGE(Qt5${M} REQUIRED HINTS ${QT_ROOT})
  IF(NOT Qt5${M}_FOUND)
    SET(ICL_ALL_QT_LIBS_FOUND FALSE)
    MESSAGE(STATUS "Not Found: Qt-Module ${M}")
  ELSE()
    MESSAGE(STATUS "Found Qt-Module ${M}")
  ENDIF()
ENDFOREACH()

IF(ICL_ALL_QT_LIBS_FOUND)
  MESSAGE(STATUS "Found: All Qt Modules")
  SET(QT_FOUND TRUE)
  add_definitions(${Qt5Widgets_DEFINITIONS})
  IF(NOT WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF(NOT WIN32)

  # This creates full library paths from Qt's colon separated paths.
  # Colon separated Qt library paths only work when Qt's find package was called.
  # This is not necessarily done in projects linking against icl so we need full
  # Paths here.
  IF(WIN32)
    SET(QWHAT IMPLIB)
    SET(QPFX Abs)
  ELSEIF(APPLE)
    SET(QWHAT LOCATION)
    SET(QPFX Abs)
  ELSE() # unix case
    SET(QWHAT SONAME)
    SET(QPFX "")
  ENDIF()

  FOREACH(M ${ICL_QT_MODULES})
    get_target_property(ICLQt5${M}Lib${QPFX} Qt5::${M} IMPORTED_${QWHAT}_RELEASE)
  ENDFOREACH()

  IF(UNIX AND NOT APPLE)
    SET(ICL_FILE_NAME_COMPONENT_DIR PATH)
    IF(CMAKE_VERSION VERSION_LESS 2.8.12)
      SET(ICL_FILE_NAME_COMPONENT_DIR DIRECTORY)
    ENDIF()
    # For some reason cmake does not add library paths to rpath when they end
    # with a number like libQt5Core.so.5. Changing the linked libraries to the
    # simple .so resolves this problem.
    get_target_property(_QT5CORE_LOCATION Qt5::Core LOCATION)
    get_filename_component(_QT5CORE_ABSOLUTE ${_QT5CORE_LOCATION} ${ICL_FILE_NAME_COMPONENT_DIR})

    FOREACH(M ${ICL_QT_MODULES})
      STRING(REGEX REPLACE "[.]so([.][0-9])+" ".so" ICLQt5${M}LibAbs "${_QT5CORE_ABSOLUTE}/${ICLQt5${M}Lib}")
    ENDFOREACH()
  ENDIF()
  SET(Qt5_LIBRARIES "")
  FOREACH(M ${ICL_QT_MODULES})
    LIST(APPEND Qt5_LIBRARIES ${ICLQt5${M}LibAbs})
  ENDFOREACH()
  add_definitions(-DICL_HAVE_QT)
ENDIF()

include_directories(${Qt5Multimedia_INCLUDE_DIRS}
                    ${Qt5Widgets_INCLUDE_DIRS}
                    ${Qt5OpenGL_INCLUDE_DIRS}
                    ${Qt5Gui_INCLUDE_DIRS}
                    ${Qt5Core_INCLUDE_DIRS})

catkin_simple(ALL_DEPS_REQUIRED)

cs_add_executable(${PROJECT_NAME}_node
  src/depth_segmentation_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node ${Qt5_LIBRARIES})

cs_install()
cs_export()
