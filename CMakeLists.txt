cmake_minimum_required(VERSION 3.0.2)
project(yolos_cpp)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  vision_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
)

## Find system dependencies
find_package(OpenCV REQUIRED)

## ONNX Runtime configuration
option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
if(NOT ONNXRUNTIME_DIR)
  # Try to find ONNX Runtime in common locations
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime/core/session/onnxruntime_cxx_api.h
    PATHS
      /usr/local/include
      /usr/include
      $ENV{HOME}/onnxruntime/include
      ${CMAKE_SOURCE_DIR}/../onnxruntime/include
      /root/work/robot_ws/src/DisableKiller/lib/onnxruntime-linux-x64-1.20.1/include
  )
  if(ONNXRUNTIME_INCLUDE_DIR)
    get_filename_component(ONNXRUNTIME_DIR "${ONNXRUNTIME_INCLUDE_DIR}/.." ABSOLUTE)
  endif()
endif()

if(ONNXRUNTIME_DIR)
  message(STATUS "ONNX Runtime directory: ${ONNXRUNTIME_DIR}")
else()
  # Try to find ONNX Runtime in DisableKiller/lib as fallback
  find_path(DISABLEKILLER_ONNX_INCLUDE
    NAMES onnxruntime_cxx_api.h
    PATHS /root/work/robot_ws/src/DisableKiller/lib
    PATH_SUFFIXES onnxruntime-linux-x64-1.20.1/include
  )
  if(DISABLEKILLER_ONNX_INCLUDE)
    get_filename_component(ONNXRUNTIME_DIR "${DISABLEKILLER_ONNX_INCLUDE}/.." ABSOLUTE)
    message(STATUS "Found ONNX Runtime in DisableKiller/lib: ${ONNXRUNTIME_DIR}")
  else()
    message(WARNING "ONNX Runtime directory not found. Please set ONNXRUNTIME_DIR or install ONNX Runtime.")
  endif()
endif()

## Catkin package configuration
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES yolos_cpp
  CATKIN_DEPENDS roscpp std_msgs std_srvs sensor_msgs vision_msgs tf2 tf2_ros tf2_geometry_msgs cv_bridge image_transport
)

## Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRUNTIME_DIR)
  include_directories("${ONNXRUNTIME_DIR}/include")
endif()


## Declare a C++ executable for the ROS YOLO detection node
add_executable(${PROJECT_NAME}_node src/yolo_detection_node.cpp)

## Specify libraries to link executable against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Link ONNX Runtime if available
if(UNIX)
    if(ONNXRUNTIME_DIR)
        target_link_libraries(${PROJECT_NAME}_node "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    else()
        # Try common ONNX Runtime lib locations
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS
              /root/work/robot_ws/src/DisableKiller/lib/onnxruntime-linux-x64-1.20.1/lib
                /usr/local/lib
                /usr/lib
        )
        if(ONNXRUNTIME_LIBRARY)
            target_link_libraries(${PROJECT_NAME}_node ${ONNXRUNTIME_LIBRARY})
            message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
        else()
            message(WARNING "ONNX Runtime library not found. Please install ONNX Runtime or set ONNXRUNTIME_DIR.")
        endif()
    endif()
endif(UNIX)

if(APPLE AND ONNXRUNTIME_DIR)
    target_link_libraries(${PROJECT_NAME}_node "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE AND ONNXRUNTIME_DIR)

if(WIN32 AND ONNXRUNTIME_DIR)
    target_link_libraries(${PROJECT_NAME}_node "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32 AND ONNXRUNTIME_DIR)

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  launch
  config
  models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
