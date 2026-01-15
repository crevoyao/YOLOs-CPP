/**
 * @file camera_inference.cpp
 * @brief ROS-based real-time object detection using YOLO models (v5, v7, v8, v9, v10, v11, v12) with camera input.
 *
 * This file serves as the main entry point for a ROS node that performs real-time object detection
 * using YOLO (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12.
 * The node subscribes to ROS camera topics, processes frames to detect objects, and
 * publishes detection results as ROS messages.
 *
 * The program operates in a ROS environment, featuring the following components:
 * 1. **ROS Subscriber**: Subscribes to camera image topics using image_transport
 * 2. **YOLO Detector**: Processes images using the specified YOLO model for object detection
 * 3. **ROS Publishers**: Publishes detection results, bounding boxes, and debug images
 * 4. **Service Interface**: Provides services to enable/disable detection and configure parameters
 *
 * Configuration parameters can be adjusted via ROS parameters:
 * - `isGPU`: Set to true to enable GPU processing for improved performance
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset)
 * - `modelPath`: Path to the desired YOLO model file (e.g., ONNX format)
 * - `enable_detection`: Enable/disable object detection
 * - `processing_rate`: Rate at which to process images (Hz)
 *
 * Debugging messages can be enabled by defining the `DEBUG_MODE` macro, allowing
 * developers to trace the execution flow and internal state of the application
 * during runtime.
 *
 * Usage Instructions:
 * 1. Launch the ROS node with appropriate parameters
 * 2. Subscribe to published topics for detection results
 * 3. Use services to control detection behavior
 *
 * @note Ensure that the required model files and labels are present in the
 * specified paths before running the application.
 *
 * Modified for ROS integration based on gesture_detection pattern
 * Original Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * ROS Integration: Following gesture_detection structure
 * Date: 29.09.2024
 */


#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <signal.h>

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// Uncomment the version
//#define YOLO5
//#define YOLO7
//#define YOLO8
//#define YOLO9
//#define YOLO10
//#define YOLO11
#define YOLO12

#ifdef YOLO5
    #include "det/YOLO5.hpp"
#endif
#ifdef YOLO7
    #include "det/YOLO7.hpp"
#endif
#ifdef YOLO8
    #include "det/YOLO8.hpp"
#endif
#ifdef YOLO9
    #include "det/YOLO9.hpp"
#endif
#ifdef YOLO10
    #include "det/YOLO10.hpp"
#endif
#ifdef YOLO11
    #include "det/YOLO11.hpp"
#endif
#ifdef YOLO12
    #include "det/YOLO12.hpp"
#endif

// Include the bounded queue
#include "tools/BoundedThreadSafeQueue.hpp"

// Global flag for graceful shutdown
volatile bool g_running = true;

void signalHandler(int signal) {
    ROS_INFO("Received signal %d, initiating shutdown...", signal);
    g_running = false;
    ros::shutdown();
}

int main()
{
    // Configuration parameters
    const bool isGPU = true;
    const std::string labelsPath = "../models/coco.names";

    #ifdef YOLO5
        std::string modelPath = "../models/yolo5-n6.onnx";
    #endif
    #ifdef YOLO7
        const std::string modelPath = "../models/yolo7-tiny.onnx";
    #endif
    #ifdef YOLO8
        std::string modelPath = "../models/yolo8n.onnx";
    #endif
    #ifdef YOLO9
        const std::string modelPath = "../models/yolov9s.onnx";
    #endif
    #ifdef YOLO10
        std::string modelPath = "../models/yolo10n_uint8.onnx";
    #endif
    #ifdef YOLO11
        const std::string modelPath = "../models/yolo11n.onnx";
    #endif
    #ifdef YOLO12
        const std::string modelPath = "../models/yolo12n.onnx";
    #endif





    const std::string videoSource = "/dev/video0"; // your usb cam device

    // Initialize YOLO detector
    #ifdef YOLO5
        YOLO5Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO7
        YOLO7Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO8
        YOLO8Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO9
        YOLO9Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO10
        YOLO10Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO11
        YOLO11Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO12
        YOLO12Detector detector(modelPath, labelsPath, isGPU);
    #endif


    // Open video capture
    cv::VideoCapture cap;
    cap.open(videoSource, cv::CAP_V4L2); // Specify V4L2 backend for better performance
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }

    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Initialize queues with bounded capacity
    const size_t max_queue_size = 2; // Double buffering
    BoundedThreadSafeQueue<cv::Mat> frameQueue(max_queue_size);
    BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Detection>>> processedQueue(max_queue_size);
    std::atomic<bool> stopFlag(false);

    // Producer thread: Capture frames
    std::thread producer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && cap.read(frame))
        {
            if (!frameQueue.enqueue(frame))
                break; // Queue is finished
        }
        frameQueue.set_finished();
    });

    // Consumer thread: Process frames
    std::thread consumer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && frameQueue.dequeue(frame))
        {
            // Perform detection
            std::vector<Detection> detections = detector.detect(frame);

            // Enqueue processed frame
            if (!processedQueue.enqueue(std::make_pair(frame, detections)))
                break;
        }
        processedQueue.set_finished();
    });

    std::pair<cv::Mat, std::vector<Detection>> item;

    #ifdef __APPLE__
    // For macOS, ensure UI runs on the main thread
    while (!stopFlag.load() && processedQueue.dequeue(item))
    {
        cv::Mat displayFrame = item.first;
        detector.drawBoundingBoxMask(displayFrame, item.second);

        cv::imshow("Detections", displayFrame);
        if (cv::waitKey(1) == 'q')
        {
            stopFlag.store(true);
            frameQueue.set_finished();
            processedQueue.set_finished();
            break;
        }
    }
    #else
    // Display thread: Show processed frames
    std::thread displayThread([&]() {
        while (!stopFlag.load() && processedQueue.dequeue(item))
        {
            cv::Mat displayFrame = item.first;
            // detector.drawBoundingBox(displayFrame, item.second);
            detector.drawBoundingBoxMask(displayFrame, item.second);

            // Display the frame
            cv::imshow("Detections", displayFrame);
            // Use a small delay and check for 'q' key press to quit
            if (cv::waitKey(1) == 'q') {
                stopFlag.store(true);
                frameQueue.set_finished();
                processedQueue.set_finished();
                break;
            }
        }
    });
    displayThread.join();
    #endif

    // Join all threads
    producer.join();
    consumer.join();

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
