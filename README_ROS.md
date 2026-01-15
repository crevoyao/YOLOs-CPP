# YOLOs-CPP ROS Package

This ROS package provides real-time object detection capabilities using the YOLOs-CPP library. It integrates YOLO (You Only Look Once) models into ROS, following the same architectural pattern as the gesture_detection package.

## Features

- **Multiple YOLO Versions**: Support for YOLOv5, v7, v8, v9, v10, v11, and v12
- **ROS Integration**: Full ROS node with topic subscription and publishing
- **GPU/CPU Support**: Configurable hardware acceleration
- **Thread-Safe Processing**: Uses bounded queues for smooth operation
- **Service Interface**: Enable/disable detection via ROS services
- **Debug Visualization**: Publishes images with bounding boxes
- **Configurable Parameters**: ROS parameter server integration

## Published Topics

- `/yolo_detection/detections` (`vision_msgs/Detection2DArray`): Object detection results
- `/yolo_detection/positions` (`geometry_msgs/PoseStamped`): 3D object positions in ARM coordinates
- `/yolo_detection/debug_image` (`sensor_msgs/Image`): Images with bounding boxes drawn
- `/yolo_detection/status` (`std_msgs/String`): Processing status messages

## Subscribed Topics

- `/camera/color/image_raw` (`sensor_msgs/Image`): Input color camera images (configurable)
- `/camera/depth/image_raw` (`sensor_msgs/Image`): Input depth camera images for 3D positioning

## Services

- `/yolo_detection/enable` (`std_srvs/SetBool`): Enable/disable object detection

## Parameters

### YOLO Model Configuration
- `model_path` (string): Path to YOLO ONNX model file
- `labels_path` (string): Path to class labels file
- `isGPU` (bool): Use GPU for inference (default: true)

### ROS Configuration
- `camera_topic` (string): Camera image topic to subscribe to
- `camera_frame` (string): Camera frame ID for published messages
- `enable_detection` (bool): Enable/disable detection (default: true)
- `processing_rate` (double): Processing rate in Hz (default: 30.0)

### 3D Positioning Configuration
- `camera_to_arm_translation` (array): Translation vector [x,y,z] from camera to ARM coordinates (meters) - **same as gesture_detection**
- `camera_to_arm_rotation` (matrix): Rotation matrix (3x3) from camera to ARM coordinates - **same as gesture_detection**

## Installation

1. **Clone YOLOs-CPP Repository**:
   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/Geekgineer/YOLOs-CPP.git
   cd YOLOs-CPP
   ```

2. **Install ONNX Runtime**:
   - Download ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases
   - Extract to a directory (e.g., `~/onnxruntime`)
   - Or set `ONNXRUNTIME_DIR` environment variable

3. **Install Dependencies**:
   ```bash
   sudo apt-get install libopencv-dev
   ```

4. **Download YOLO Models**:
   - Download models from the [YOLOs-CPP Mega Drive](https://mega.nz/folder/TvgXVRQJ#6M0IZdMOvKlKY9-dx7Uu7Q)
   - Place in the `models/` directory

5. **Build the Package**:
   ```bash
   cd ~/catkin_ws
   catkin_make --pkg yolos_cpp
   ```

   Or with ONNX Runtime path:
   ```bash
   catkin_make --pkg yolos_cpp -DONNXRUNTIME_DIR=/path/to/onnxruntime
   ```

## Usage

### Launch File

```bash
roslaunch yolos_cpp yolo_detection.launch
```

### With Parameters

```bash
roslaunch yolos_cpp yolo_detection.launch \
  camera_topic:=/my_camera/image_raw \
  model_path:=$(find yolos_cpp)/models/yolo8n.onnx \
  is_gpu:=false
```

### Direct Node Execution

```bash
rosrun yolos_cpp yolos_cpp_node \
  _camera_topic:=/camera/color/image_raw \
  _model_path:=models/yolo11n.onnx \
  _isGPU:=true
```

## YOLO Model Selection

To use a different YOLO version, modify the compile-time selection in `src/yolo_detection_node.cpp`:

```cpp
// Uncomment the desired YOLO version
//#define YOLO5
//#define YOLO7
//#define YOLO8
//#define YOLO9
//#define YOLO10
//#define YOLO11
#define YOLO12  // Currently selected
```

Then rebuild the package:

```bash
catkin_make --pkg yolos_cpp
```

## ROS Integration Example

### Python Subscriber

```python
#!/usr/bin/env python

import rospy
from vision_msgs.msg import Detection2DArray

def detection_callback(msg):
    for detection in msg.detections:
        bbox = detection.bbox
        confidence = detection.results[0].score
        class_id = detection.results[0].id
        print(f"Detected object: class={class_id}, confidence={confidence:.2f}")
        print(f"Bounding box: center=({bbox.center.x}, {bbox.center.y}), size=({bbox.size_x}, {bbox.size_y})")

def main():
    rospy.init_node('yolo_subscriber')
    rospy.Subscriber('/yolo_detection/detections', Detection2DArray, detection_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
```

### Enable/Disable Detection

```bash
# Enable detection
rosservice call /yolo_detection/enable "data: true"

# Disable detection
rosservice call /yolo_detection/enable "data: false"
```

## Troubleshooting

### ONNX Runtime Not Found
If you get linking errors related to ONNX Runtime:

1. Set the `ONNXRUNTIME_DIR` environment variable:
   ```bash
   export ONNXRUNTIME_DIR=/path/to/onnxruntime
   ```

2. Or pass it to catkin_make:
   ```bash
   catkin_make --pkg yolos_cpp -DONNXRUNTIME_DIR=/path/to/onnxruntime
   ```

### Camera Topic Issues
- Verify your camera is publishing to the correct topic
- Use `rostopic list` to see available topics
- Use `rostopic info /camera/color/image_raw` to check topic details

### Performance Issues
- Reduce `processing_rate` parameter for slower hardware
- Set `isGPU` to `false` if GPU is not available
- Use smaller YOLO models (e.g., `yolo8n.onnx` instead of `yolo8x.onnx`)

## 3D Object Positioning

The node provides 3D positioning of detected objects in ARM coordinates, **using the exact same transformation parameters as the gesture_detection system**:

- **Translation**: [0.3391445173213441, 0.16158612387789312, -0.036710996800156034]
- **Rotation Matrix** (transposed):
  ```
  [-0.9683492, 0.24959585, -0.00131968]
  [-0.03955044, -0.15865874, -0.98654101]
  [-0.24644592, -0.955264, 0.16350869]
  ```

### How it works:
1. **Latest Depth Usage**: Uses the most recent depth image available (no timestamp synchronization required)
2. **Object Center Calculation**: Finds the center pixel of each detected bounding box
3. **Epipolar Geometry**: Uses fundamental matrix to find corresponding depth pixel
4. **Depth Filtering**: Applies median filtering for robust depth estimation
5. **3D Reconstruction**: Converts 2D pixel + depth to 3D point in camera coordinates
6. **Coordinate Transformation**: Transforms from camera to ARM coordinate system

### Published 3D Data:
- **Frame**: `j2n6s300_link_base` (Kinova robot base frame)
- **Position**: [x, y, z] in meters
- **Metadata**: Object class ID and confidence stored in pose orientation fields

## Architecture

The ROS node follows the gesture_detection pattern:

1. **Synchronization**: Receives and synchronizes color/depth image pairs
2. **Processing Thread**: Performs YOLO inference and 3D position calculation
3. **Publisher Threads**: Publishes detection results, 3D positions, and debug images

Thread-safe queues ensure smooth operation and prevent frame drops during processing.

## Dependencies

- **ROS**: Kinetic or newer
- **OpenCV**: 4.5.5 or newer
- **ONNX Runtime**: 1.16.3 or newer
- **YOLOs-CPP Library**: Included in this package

## License

This ROS package integrates the MIT-licensed YOLOs-CPP library.

## Contributing

1. Follow ROS coding standards
2. Use the existing architectural pattern
3. Test with multiple YOLO versions
4. Update documentation for any new features
