#!/usr/bin/env python3

"""
Holonomic Robot Pose Detection using ArUco Markers
Detects robot pose from overhead camera and publishes to /bot_pose
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from hb_interfaces.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
from cv2 import aruco


class PoseDetector(Node):
    """Node for detecting robot pose using ArUco markers"""
    
    def __init__(self):
        super().__init__('holonomic_perception')
        
        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()
        
        # ArUco marker detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Publisher for robot pose
        self.pose_pub = self.create_publisher(Pose2D, '/bot_pose', 10)
        
        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, 
            "/camera/image_raw", 
            self.image_callback, 
            10
        )
        
        # Previous pose for filtering
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        
        # Smoothing factor (0 = no smoothing, 1 = max smoothing)
        self.alpha = 0.3
        
        self.get_logger().info('Holonomic Perception Node Started')
        self.get_logger().info('Listening to: /camera/image_raw')
        self.get_logger().info('Publishing to: /bot_pose')
    
    def image_callback(self, msg):
        """Callback for processing camera images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect ArUco markers
            corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
            
            if ids is not None and len(ids) > 0:
                # Process detected markers
                pose = self.calculate_pose(corners, ids, cv_image.shape)
                
                if pose is not None:
                    # Publish pose
                    self.pose_pub.publish(pose)
                    
                    # Optional: Draw markers on image for visualization
                    # cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    # cv2.imshow('ArUco Detection', cv_image)
                    # cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def calculate_pose(self, corners, ids, image_shape):
        """
        Calculate robot pose from detected ArUco markers
        
        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Pose2D message with x, y, theta
        """
        # Find the robot marker (typically ID 0 for the robot base)
        robot_marker_id = 0
        robot_marker_index = None
        
        for i, marker_id in enumerate(ids):
            if marker_id[0] == robot_marker_id:
                robot_marker_index = i
                break
        
        if robot_marker_index is None:
            # Robot marker not found, return None
            return None
        
        # Get corners of robot marker
        marker_corners = corners[robot_marker_index][0]
        
        # Calculate center of marker (x, y position)
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        
        # Calculate orientation (theta) from marker corners
        # Vector from corner 0 to corner 1 gives the orientation
        dx = marker_corners[1, 0] - marker_corners[0, 0]
        dy = marker_corners[1, 1] - marker_corners[0, 1]
        theta = np.arctan2(dy, dx)
        
        # Apply exponential smoothing for stability
        if self.prev_x is not None:
            center_x = self.alpha * center_x + (1 - self.alpha) * self.prev_x
            center_y = self.alpha * center_y + (1 - self.alpha) * self.prev_y
            
            # Handle angle wrapping for theta smoothing
            angle_diff = self.normalize_angle(theta - self.prev_theta)
            theta = self.prev_theta + self.alpha * angle_diff
        
        # Update previous values
        self.prev_x = center_x
        self.prev_y = center_y
        self.prev_theta = theta
        
        # Create and return Pose2D message
        pose_msg = Pose2D()
        pose_msg.x = float(center_x)
        pose_msg.y = float(center_y)
        pose_msg.theta = float(theta)
        
        return pose_msg
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    
    try:
        pose_detector = PoseDetector()
        rclpy.spin(pose_detector)
    except KeyboardInterrupt:
        print('\nPerception node interrupted')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            pose_detector.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()