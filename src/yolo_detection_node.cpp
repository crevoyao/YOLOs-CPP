/**
 * @file yolo_detection_node.cpp
 * @brief ROS node for real-time YOLO object detection using YOLOs-CPP library
 *
 * This ROS node integrates the YOLOs-CPP library to provide real-time object detection
 * capabilities in ROS environments. It subscribes to camera image topics, performs
 * YOLO-based object detection, and publishes detection results as ROS messages.
 *
 * The node follows the same architectural pattern as gesture_detection, providing:
 * - ROS topic subscription for camera images
 * - Configurable parameters via ROS parameter server
 * - Thread-safe processing with bounded queues
 * - ROS service interfaces for control
 * - Multiple output topics for different types of results
 *
 * Features:
 * - Support for multiple YOLO versions (v5, v7, v8, v9, v10, v11, v12)
 * - GPU/CPU processing modes
 * - Configurable detection parameters
 * - Debug image publishing
 * - Status reporting
 * - Service-based enable/disable control
 *
 * Usage:
 * 1. Launch the node with appropriate parameters
 * 2. Subscribe to published detection topics
 * 3. Use services to control detection behavior
 *
 * Author: ROS Integration of YOLOs-CPP
 * Date: 2024
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <signal.h>

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <chrono>

// Uncomment the YOLO version to use
//#define YOLO5
//#define YOLO7
#define YOLO8
//#define YOLO9
//#define YOLO10
//#define YOLO11
//#define YOLO12

// Include YOLO detector headers based on selected version
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

// Include the bounded queue for thread-safe processing
#include "tools/BoundedThreadSafeQueue.hpp"

// Global flag for graceful shutdown
volatile bool g_running = true;

void signalHandler(int signal) {
    ROS_INFO("Received signal %d, initiating shutdown...", signal);
    g_running = false;

    // Request ROS shutdown for clean exit
    ros::shutdown();

    // Also print to console in case ROS logging is not working
    std::cout << "\nShutdown signal received (signal " << signal << "), exiting..." << std::endl;
}

/**
 * @class YOLODetectionNode
 * @brief Main ROS node class for YOLO object detection
 *
 * This class encapsulates all ROS functionality for YOLO-based object detection,
 * including topic subscription, parameter loading, detection processing, and result publishing.
 */
class YOLODetectionNode {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     * @param pnh Private ROS node handle for parameters
     */
    YOLODetectionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), it_(nh), tf_listener_(tf_buffer_) {

        // Load parameters from ROS parameter server
        loadParameters();

        // Initialize camera calibration matrices
        initializeCalibrationMatrices();

        // Initialize YOLO detector with loaded parameters
        initializeDetector();

        // Set up ROS communication interfaces
        initializeSubscribers();
        initializePublishers();
        initializeServices();

        // Initialize thread-safe processing queue (color and depth image pairs)
        const size_t max_queue_size = 2; // Double buffering for smooth processing
        frameQueue_ = std::make_unique<BoundedThreadSafeQueue<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>>>(max_queue_size);

        ROS_INFO("YOLO Detection Node initialized successfully");
    }

    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~YOLODetectionNode() {
        ROS_INFO("YOLODetectionNode destructor called");
        stopProcessing();
        if (processing_thread_.joinable()) {
            ROS_INFO("Waiting for processing thread to finish...");
            processing_thread_.join();
        }
        ROS_INFO("YOLODetectionNode cleanup complete");
    }

    /**
     * @brief Main run loop
     *
     * Starts processing thread and maintains ROS spin loop until shutdown
     */
    void run() {
        // Start the detection processing thread
        startProcessing();

        // Main ROS spin loop
        ros::Rate rate(processing_rate_);

        while (ros::ok()) {
            ros::spinOnce();

            // Check for shutdown signal more frequently
            if (!g_running) {
                ROS_INFO("Shutdown signal received, exiting...");
                break;
            }

            // Allow immediate exit on shutdown
            if (!ros::ok()) {
                ROS_INFO("ROS shutdown requested, exiting...");
                break;
            }

            rate.sleep();
        }

        ROS_INFO("Stopping processing thread...");
        stopProcessing();
        ROS_INFO("YOLO detection node shutdown complete");
    }

private:
    /**
     * @brief Load parameters from ROS parameter server
     *
     * Loads YOLO model parameters, ROS topics, and processing settings
     */
    void loadParameters() {
        // Load YOLO-specific parameters
        pnh_.param<bool>("isGPU", isGPU_, true);
        pnh_.param<std::string>("labelsPath", labelsPath_, "models/coco.names");

        // Load model path based on selected YOLO version
        #ifdef YOLO5
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolo5-n6.onnx");
        #endif
        #ifdef YOLO7
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolo7-tiny.onnx");
        #endif
        #ifdef YOLO8
            pnh_.param<std::string>("modelPath", modelPath_, "models/best.onnx");
        #endif
        #ifdef YOLO9
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolov9s.onnx");
        #endif
        #ifdef YOLO10
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolo10n_uint8.onnx");
        #endif
        #ifdef YOLO11
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolo11n.onnx");
        #endif
        #ifdef YOLO12
            pnh_.param<std::string>("modelPath", modelPath_, "models/yolo12n.onnx");
        #endif

        // Load ROS-specific parameters
        pnh_.param<bool>("enable_detection", enable_detection_, true);
        pnh_.param<double>("processing_rate", processing_rate_, 30.0);
        pnh_.param<std::string>("camera_topic", camera_topic_, "/camera/color/image_raw");
        pnh_.param<std::string>("camera_frame", camera_frame_, "camera_color_optical_frame");

        ROS_INFO("Parameters loaded - GPU: %s, Model: %s, Labels: %s",
                 isGPU_ ? "true" : "false", modelPath_.c_str(), labelsPath_.c_str());
    }

    /**
     * @brief Initialize camera calibration matrices
     *
     * Sets up camera intrinsics, extrinsics, and fundamental matrix for 3D conversion
     */
    void initializeCalibrationMatrices() {
        // Color camera intrinsics (from camera_transformation.cpp reference)
        color_intrinsics_ << 602.097670, 0.000000, 326.478188,
                            0.000000, 602.832999, 250.443734,
                            0.000000, 0.000000, 1.000000;

        // Depth camera intrinsics (from camera_transformation.cpp reference)
        depth_intrinsics_ << 577.981205, 0.000000, 315.511467,
                            0.000000, 579.607259, 244.899492,
                            0.000000, 0.000000, 1.000000;

        // Depth to color transformation (from camera_transformation.cpp reference)
        Eigen::Vector3d depth_to_color_translation(-0.02458872, 0.00112017, -0.0005721);
        Eigen::Matrix3d depth_to_color_rotation;
        depth_to_color_rotation << 0.9994915, 0.02805794, -0.0151491,
                                 -0.0279967, 0.99959903, 0.00423949,
                                 0.01526197, -0.00381321, 0.99987626;

        // Compute fundamental matrix F = K_color^-T * [t]×R * K_depth^-1
        Eigen::Matrix3d color_intrinsics_inv = color_intrinsics_.inverse();
        Eigen::Matrix3d depth_intrinsics_inv = depth_intrinsics_.inverse();

        Eigen::Matrix3d skew_t;
        skew_t << 0, -depth_to_color_translation.z(), depth_to_color_translation.y(),
                  depth_to_color_translation.z(), 0, -depth_to_color_translation.x(),
                  -depth_to_color_translation.y(), depth_to_color_translation.x(), 0;

        Eigen::Matrix3d essential = skew_t * depth_to_color_rotation;
        Eigen::Matrix3d fundamental_eigen = color_intrinsics_inv.transpose() * essential * depth_intrinsics_inv;

        // Convert Eigen fundamental matrix to OpenCV format
        F_ = (cv::Mat_<double>(3, 3) <<
              fundamental_eigen(0,0), fundamental_eigen(0,1), fundamental_eigen(0,2),
              fundamental_eigen(1,0), fundamental_eigen(1,1), fundamental_eigen(1,2),
              fundamental_eigen(2,0), fundamental_eigen(2,1), fundamental_eigen(2,2));

        // Camera to ARM transformation (same as gesture_detection main.cpp)
        // Load camera to ARM transformation from parameters with defaults from gesture_detection
        std::vector<double> camera_to_arm_trans, camera_to_arm_rot;
        pnh_.param("camera_to_arm_translation", camera_to_arm_trans,
                  std::vector<double>{0.3391445173213441, 0.16158612387789312, -0.036710996800156034});
        pnh_.param("camera_to_arm_rotation", camera_to_arm_rot,
                  std::vector<double>{-0.9683492, 0.24959585, -0.00131968,
                                    -0.03955044, -0.15865874, -0.98654101,
                                    -0.24644592, -0.955264, 0.16350869});

        // Validate parameter sizes
        if (camera_to_arm_trans.size() == 3 && camera_to_arm_rot.size() == 9) {
            camera_to_arm_translation_ << camera_to_arm_trans[0], camera_to_arm_trans[1], camera_to_arm_trans[2];
            camera_to_arm_rotation_ << camera_to_arm_rot[0], camera_to_arm_rot[1], camera_to_arm_rot[2],
                                      camera_to_arm_rot[3], camera_to_arm_rot[4], camera_to_arm_rot[5],
                                      camera_to_arm_rot[6], camera_to_arm_rot[7], camera_to_arm_rot[8];

            ROS_INFO("Loaded camera to ARM transformation: translation=(%.3f, %.3f, %.3f)",
                     camera_to_arm_translation_.x(), camera_to_arm_translation_.y(), camera_to_arm_translation_.z());
        } else {
            ROS_ERROR("Invalid camera to ARM transformation parameters - translation size: %lu, rotation size: %lu",
                     camera_to_arm_trans.size(), camera_to_arm_rot.size());
            // Use identity as fallback
            camera_to_arm_rotation_ = Eigen::Matrix3d::Identity();
            camera_to_arm_translation_ = Eigen::Vector3d::Zero();
        }

        ROS_INFO("Camera calibration matrices initialized");
    }

    /**
     * @brief Initialize YOLO detector
     *
     * Creates and initializes the appropriate YOLO detector based on compile-time selection
     */
    void initializeDetector() {
        try {
            // Create detector instance based on selected YOLO version
            #ifdef YOLO5
                detector_ = std::make_unique<YOLO5Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv5 detector");
            #endif
            #ifdef YOLO7
                detector_ = std::make_unique<YOLO7Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv7 detector");
            #endif
            #ifdef YOLO8
                detector_ = std::make_unique<YOLO8Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv8 detector");
            #endif
            #ifdef YOLO9
                detector_ = std::make_unique<YOLO9Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv9 detector");
            #endif
            #ifdef YOLO10
                detector_ = std::make_unique<YOLO10Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv10 detector");
            #endif
            #ifdef YOLO11
                detector_ = std::make_unique<YOLO11Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv11 detector");
            #endif
            #ifdef YOLO12
                detector_ = std::make_unique<YOLO12Detector>(modelPath_, labelsPath_, isGPU_);
                ROS_INFO("Initialized YOLOv12 detector");
            #endif

            ROS_INFO("YOLO detector initialized successfully");
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to initialize YOLO detector: %s", e.what());
            throw;
        }
    }

    /**
     * @brief Initialize ROS subscribers
     */
    void initializeSubscribers() {
        // Subscribe to color and depth images
        color_sub_ = it_.subscribe(camera_topic_, 1,
                                  &YOLODetectionNode::colorCallback, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
                                  &YOLODetectionNode::depthCallback, this);
        ROS_INFO("Subscribed to camera topics: %s and /camera/depth/image_raw", camera_topic_.c_str());
    }

    /**
     * @brief Initialize ROS publishers
     */
    void initializePublishers() {
        // Publisher for detection results in vision_msgs format
        detection_pub_ = nh_.advertise<vision_msgs::Detection2DArray>("/yolo_detection/detections", 10);

        // Publisher for 3D object positions in ARM coordinates
        position_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/yolo_detection/positions", 10);

        // Publisher for debug images with bounding boxes
        debug_image_pub_ = it_.advertise("/yolo_detection/debug_image", 1);

        // Publisher for status messages
        status_pub_ = nh_.advertise<std_msgs::String>("/yolo_detection/status", 10);

        ROS_INFO("Initialized ROS publishers");
    }

    /**
     * @brief Initialize ROS services
     */
    void initializeServices() {
        // Service to enable/disable detection
        enable_service_ = nh_.advertiseService("/yolo_detection/enable",
                                              &YOLODetectionNode::enableDetectionService, this);
        ROS_INFO("Initialized ROS services");
    }

    /**
     * @brief Color camera image callback
     * @param msg Incoming ROS color image message
     */
    void colorCallback(const sensor_msgs::ImageConstPtr& msg) {
        if (!enable_detection_) {
            return; // Skip processing if detection is disabled
        }

        // Get the latest depth image (no synchronization needed)
        sensor_msgs::ImageConstPtr depth_msg;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            depth_msg = latest_depth_image_;
        }

        // If we don't have a depth image yet, skip processing
        if (!depth_msg) {
            ROS_WARN_THROTTLE(1.0, "No depth image available yet, skipping detection");
            return;
        }

        // Enqueue image pair for processing (color image + latest depth image)
        auto image_pair = std::make_pair(msg, depth_msg);
        if (!frameQueue_->enqueue(image_pair)) {
            ROS_WARN_THROTTLE(1.0, "Frame queue full, dropping frame pair");
        }
    }

    /**
     * @brief Depth camera image callback
     * @param msg Incoming ROS depth image message
     */
    void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        latest_depth_image_ = msg;  // Just store the latest depth image
    }

    /**
     * @brief Start the processing thread
     */
    void startProcessing() {
        processing_thread_ = std::thread(&YOLODetectionNode::processingLoop, this);
        ROS_INFO("Started detection processing thread");
    }

    /**
     * @brief Stop the processing thread
     */
    void stopProcessing() {
        stop_flag_ = true;
        frameQueue_->set_finished();
    }

    /**
     * @brief Main processing loop (runs in separate thread)
     *
     * Continuously processes synchronized color-depth image pairs and publishes results
     */
    void processingLoop() {
        std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr> image_pair;

        // Process image pairs until shutdown
        while (!stop_flag_ && g_running && ros::ok()) {
            // Try to dequeue an image pair (non-blocking check for shutdown)
            if (!frameQueue_->dequeue(image_pair)) {
                // No image available, sleep briefly to prevent busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            try {
                auto color_msg = image_pair.first;
                auto depth_msg = image_pair.second;

                // Convert ROS images to OpenCV format
                cv_bridge::CvImageConstPtr color_cv_ptr = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8);
                cv_bridge::CvImageConstPtr depth_cv_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);

                // Perform YOLO object detection on color image
                std::vector<Detection> detections = detector_->detect(color_cv_ptr->image);

                // Publish detection results
                publishDetections(detections, color_msg->header);

                // Publish 3D positions in ARM coordinates
                publishObjectPositions(detections, color_cv_ptr->image, depth_cv_ptr->image, color_msg->header);

                // Publish debug image with bounding boxes
                publishDebugImage(color_cv_ptr->image, detections, color_msg->header);

                // Publish processing status
                std_msgs::String status_msg;
                status_msg.data = "Detection completed: " + std::to_string(detections.size()) + " objects detected";
                status_pub_.publish(status_msg);

                ROS_DEBUG("Processed frame with %lu detections", detections.size());

            } catch (const cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
            } catch (const std::exception& e) {
                ROS_ERROR("Detection processing error: %s", e.what());
            }
        }

        ROS_INFO("Detection processing thread finished");
    }

    /**
     * @brief Publish detection results
     * @param detections Vector of detected objects
     * @param header ROS message header from original image
     *
     * Converts YOLO detections to ROS vision_msgs format and publishes
     */
    void publishDetections(const std::vector<Detection>& detections, const std_msgs::Header& header) {
        vision_msgs::Detection2DArray detection_array;
        detection_array.header = header;
        detection_array.header.frame_id = camera_frame_;

        // Convert each detection to ROS message format
        for (const auto& det : detections) {
            vision_msgs::Detection2D detection;
            detection.header = header;
            detection.header.frame_id = camera_frame_;

            // Set bounding box (center and size format)
            detection.bbox.center.x = det.box.x + det.box.width / 2.0;
            detection.bbox.center.y = det.box.y + det.box.height / 2.0;
            detection.bbox.size_x = det.box.width;
            detection.bbox.size_y = det.box.height;

            // Set detection hypothesis (class and confidence)
            vision_msgs::ObjectHypothesisWithPose hypothesis;
            hypothesis.id = det.classId;
            hypothesis.score = det.conf;
            detection.results.push_back(hypothesis);

            detection_array.detections.push_back(detection);
        }

        detection_pub_.publish(detection_array);
    }

    /**
     * @brief Publish 3D positions of detected objects in ARM coordinates
     * @param detections Vector of detected objects
     * @param color_image Color image for center calculation
     * @param depth_image Depth image for depth values
     * @param header ROS message header
     */
    void publishObjectPositions(const std::vector<Detection>& detections,
                               const cv::Mat& color_image,
                               const cv::Mat& depth_image,
                               const std_msgs::Header& header) {
        for (const auto& det : detections) {
            // Calculate object center in color image
            cv::Point2d object_center(det.box.x + det.box.width / 2.0,
                                    det.box.y + det.box.height / 2.0);

            // Convert to 3D ARM coordinates
            Eigen::Vector3d arm_position = colorPixelTo3DARM(object_center, depth_image);

            if (arm_position.isZero()) {
                ROS_WARN("Could not compute 3D position for object at pixel (%.1f, %.1f)",
                        object_center.x, object_center.y);
                continue;
            }

            // Create pose message
            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header = header;
            pose_msg.header.frame_id = "j2n6s300_link_base";  // ARM base frame
            pose_msg.pose.position.x = arm_position.x();
            pose_msg.pose.position.y = arm_position.y();
            pose_msg.pose.position.z = arm_position.z();


            tf2::Quaternion q0(0.707, 0.0, 0.0, 0.707);   // (x,y,z,w) hand point towards -Y
            q0.normalize();

            double theta = M_PI / 6.0;  // 30 deg rotate z axis +30

            tf2::Quaternion qz;
            qz.setRPY(0.0, 0.0, theta);  // rotation about Z

            // Apply world-frame Z rotation
            tf2::Quaternion q = q0 * qz;
            q.normalize();
            pose_msg.pose.orientation = tf2::toMsg(q0);
            

            position_pub_.publish(pose_msg);

            ROS_DEBUG("Published 3D position for object class %d at (%.3f, %.3f, %.3f)",
                     det.classId, arm_position.x(), arm_position.y(), arm_position.z());
        }
    }

    /**
     * @brief Convert color pixel to 3D ARM coordinates
     * @param color_pixel 2D pixel coordinates in color image
     * @param depth_image Depth image for depth values
     * @return 3D position in ARM coordinates
     *
     * References gesture_detection depth_processor.cpp colorPixelTo3DPointARM method
     */
    Eigen::Vector3d colorPixelTo3DARM(cv::Point2d& color_pixel, const cv::Mat& depth_image) {
        std::lock_guard<std::mutex> lock(processing_mutex_);

        // Find corresponding depth pixel using epipolar geometry
        cv::Point2d depth_pixel = colorPixelToDepthPixelEpipolar(color_pixel, F_);

        // Get depth using median filtering around the corresponding depth pixel
        double depth = getDepthMedian(depth_image,
                                    static_cast<int>(depth_pixel.x),
                                    static_cast<int>(depth_pixel.y),
                                    3); // 3x3 neighborhood

        if (depth <= 0) {
            ROS_WARN("Invalid median depth at pixel (%.1f, %.1f) for color pixel (%.1f, %.1f)",
                    depth_pixel.x, depth_pixel.y, color_pixel.x, color_pixel.y);
            return Eigen::Vector3d::Zero();
        }
  
        // Convert color pixel to 3D point in color camera coordinates
        Eigen::Vector3d color_camera_point = backproject2DTo3D(color_pixel, depth, color_intrinsics_);
        // Transform from color camera to ARM coordinates
        Eigen::Vector3d arm_point = camera_to_arm_rotation_ * color_camera_point + camera_to_arm_translation_;

        ROS_INFO("Converted color pixel (%.1f, %.1f) -> depth pixel (%.1f, %.1f) -> ARM (%.3f, %.3f, %.3f) with median depth %.3f",
                 color_pixel.x, color_pixel.y, depth_pixel.x, depth_pixel.y,
                 arm_point.x(), arm_point.y(), arm_point.z(), depth);

        return arm_point;
    }

    /**
     * @brief Convert color pixel to depth pixel using epipolar geometry
     * @param color_pixel 2D pixel in color image
     * @param fundamental_matrix Fundamental matrix between color and depth cameras
     * @return Corresponding pixel in depth image
     */
    cv::Point2d colorPixelToDepthPixelEpipolar(const cv::Point2d& color_pixel, const cv::Mat& fundamental_matrix) {
        // Convert color pixel to homogeneous coordinates
        cv::Mat ur_h = (cv::Mat_<double>(3,1) << color_pixel.x, color_pixel.y, 1.0);

        // Compute epipolar line in depth image: l = F^T * ur_h
        cv::Mat l = fundamental_matrix.t() * ur_h;
        cv::Vec3d line(l.at<double>(0,0), l.at<double>(1,0), l.at<double>(2,0));

        // Project the color pixel (with same x,y) onto the epipolar line
        cv::Point2d sameXY(color_pixel.x, color_pixel.y);
        cv::Point2d corresponding_depth_pixel = projectPointToLine(line, sameXY);

        return corresponding_depth_pixel;
    }

    /**
     * @brief Project a point onto an epipolar line
     * @param line Epipolar line coefficients (a, b, c) where ax + by + c = 0
     * @param p0 Reference point
     * @return Point on the line closest to p0
     */
    cv::Point2d projectPointToLine(const cv::Vec3d& line, const cv::Point2d& p0) {
        double a = line[0], b = line[1], c = line[2];
        double denom = a*a + b*b;
        if (denom == 0) return p0;
        double k = (a * p0.x + b * p0.y + c) / denom;
        double x = p0.x - a * k;
        double y = p0.y - b * k;
        return cv::Point2d(x, y);
    }

    /**
     * @brief Backproject 2D pixel to 3D point
     * @param point_2d 2D pixel coordinates
     * @param depth Depth value
     * @param intrinsics Camera intrinsic matrix
     * @return 3D point in camera coordinates
     */
    Eigen::Vector3d backproject2DTo3D(const cv::Point2d& point_2d, double depth, const Eigen::Matrix3d& intrinsics) {
        Eigen::Vector3d point_2d_homogeneous(point_2d.x, point_2d.y, 1.0);
        Eigen::Vector3d ray = intrinsics.inverse() * point_2d_homogeneous;
        return ray * depth;
    }

    /**
     * @brief Get median depth value around a pixel
     * @param depth_image Depth image
     * @param center_x Center pixel x coordinate
     * @param center_y Center pixel y coordinate
     * @param window_size Size of median filter window (odd number)
     * @return Median depth value
     */
    double getDepthMedian(const cv::Mat& depth_image, int center_x, int center_y, int window_size) {
        std::vector<float> depths;

        int half_window = window_size / 2;
        for (int dy = -half_window; dy <= half_window; ++dy) {
            for (int dx = -half_window; dx <= half_window; ++dx) {
                int x = center_x + dx;
                int y = center_y + dy;

                if (x >= 0 && x < depth_image.cols && y >= 0 && y < depth_image.rows) {
                    float depth = depth_image.at<float>(y, x) / 1000.0;
                    if (depth > 0 && std::isfinite(depth)) {
                        depths.push_back(depth);
                    }
                }
            }
        }

        if (depths.empty()) {
            return 0.0;
        }

        // Sort and return median
        std::sort(depths.begin(), depths.end());
        size_t mid = depths.size() / 2;
        return depths.size() % 2 == 0 ? (depths[mid-1] + depths[mid]) / 2.0 : depths[mid];
    }

    /**
     * @brief Publish debug image with detections
     * @param image Original OpenCV image
     * @param detections Vector of detected objects
     * @param header ROS message header
     *
     * Creates and publishes a debug image with bounding boxes drawn
     */
    void publishDebugImage(const cv::Mat& image, const std::vector<Detection>& detections,
                          const std_msgs::Header& header) {
        // Create a copy of the image for drawing
        cv::Mat debug_image = image.clone();

        // Draw bounding boxes and labels using YOLO detector's drawing function
        detector_->drawBoundingBoxMask(debug_image, detections);

        // Convert back to ROS image message
        cv_bridge::CvImage debug_bridge;
        debug_bridge.header = header;
        debug_bridge.header.frame_id = camera_frame_;
        debug_bridge.encoding = sensor_msgs::image_encodings::BGR8;
        debug_bridge.image = debug_image;

        debug_image_pub_.publish(debug_bridge.toImageMsg());
    }

    /**
     * @brief Service callback to enable/disable detection
     * @param req Service request
     * @param res Service response
     * @return True if service call successful
     */
    bool enableDetectionService(std_srvs::SetBool::Request& req,
                               std_srvs::SetBool::Response& res) {
        enable_detection_ = req.data;
        res.success = true;
        res.message = enable_detection_ ? "Object detection enabled" : "Object detection disabled";
        ROS_INFO("Object detection %s", enable_detection_ ? "enabled" : "disabled");
        return true;
    }

    // ROS communication interfaces
    ros::NodeHandle nh_;           // Node handle
    ros::NodeHandle pnh_;          // Private node handle for parameters
    image_transport::ImageTransport it_;  // Image transport for compressed images

    // ROS subscribers
    image_transport::Subscriber color_sub_;  // Color camera image subscriber
    image_transport::Subscriber depth_sub_;  // Depth camera image subscriber

    // ROS publishers
    ros::Publisher detection_pub_;           // Detection results publisher
    ros::Publisher position_pub_;            // 3D position publisher
    ros::Publisher status_pub_;              // Status messages publisher
    image_transport::Publisher debug_image_pub_;  // Debug images publisher

    // ROS services
    ros::ServiceServer enable_service_;      // Enable/disable service

    // TF2
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Thread safety
    std::mutex depth_mutex_;
    std::mutex processing_mutex_;

    // Latest depth image (for 3D processing)
    sensor_msgs::ImageConstPtr latest_depth_image_;

    // Processing
    std::unique_ptr<BoundedThreadSafeQueue<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>>> frameQueue_;  // Thread-safe frame pair queue
    std::thread processing_thread_;           // Detection processing thread
    std::atomic<bool> stop_flag_{false};     // Thread stop flag

    // YOLO detector (selected at compile time)
    #ifdef YOLO5
        std::unique_ptr<YOLO5Detector> detector_;
    #endif
    #ifdef YOLO8
        std::unique_ptr<YOLO8Detector> detector_;
    #endif

    // Camera calibration matrices
    Eigen::Matrix3d color_intrinsics_;      // Color camera intrinsic matrix
    Eigen::Matrix3d depth_intrinsics_;      // Depth camera intrinsic matrix
    cv::Mat F_;                             // Fundamental matrix for epipolar geometry
    Eigen::Matrix3d camera_to_arm_rotation_; // Rotation from camera to ARM coordinates
    Eigen::Vector3d camera_to_arm_translation_; // Translation from camera to ARM coordinates

    // Configuration parameters
    bool isGPU_;                    // Use GPU for inference
    std::string labelsPath_;        // Path to class labels file
    std::string modelPath_;         // Path to YOLO model file
    bool enable_detection_;         // Enable/disable detection flag
    double processing_rate_;        // Processing rate (Hz)
    std::string camera_topic_;      // Camera image topic name
    std::string camera_frame_;      // Camera frame ID
};

/**
 * @brief Main function
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return Exit code
 */
int main(int argc, char** argv) {
    // Initialize ROS node (let ROS handle signals, but add backup handler)
    ros::init(argc, argv, "yolo_detection_node");

    // Add backup signal handler in case ROS signal handling fails
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Create ROS node handles
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try {
        // Create and run the YOLO detection node
        YOLODetectionNode node(nh, pnh);
        node.run();
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize YOLO detection node: %s", e.what());
        return -1;
    } catch (...) {
        ROS_ERROR("Unknown exception occurred in YOLO detection node");
        return -1;
    }

    ROS_INFO("YOLO detection node shutdown complete");
    return 0;
}
