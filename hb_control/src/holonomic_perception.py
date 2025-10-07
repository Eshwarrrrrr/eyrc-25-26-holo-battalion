#!/usr/bin/env python3
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D

class HolonomicPoseDetector(Node):
    def __init__(self):
        super().__init__('holonomic_localization_node')
        self.bridge = CvBridge()
        self.crate_marker_size = 0.05
        self.bot_marker_size = 0.05
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None
            self.use_new_api = False

        self.bot_ids = [8, 9]
        self.corner_ids = [1, 3, 7, 5]
        self.world_coords = {1:[0,0], 3:[2438.4,0], 7:[2438.4,2438.4], 5:[0,2438.4]}
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self.camera_info_callback, 10)
        self.bot_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.crate_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_ready = False
        self.pixel_matrix = []
        self.world_matrix = []
        self.H = None
        self.prev_yaw = {}

    def camera_info_callback(self, msg):
        if not self.camera_ready:
            self.camera_matrix = np.array(msg.k).reshape(3,3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_ready = True

    def pixel_to_world(self, px, py):
        if self.H is None:
            return None, None
        pt = np.array([[[px, py]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.H)
        return float(world_pt[0][0][0]), float(world_pt[0][0][1])

    def compute_yaw(self, corners):
        c = corners[0]
        dx = c[1][0]-c[0][0]
        dy = c[1][1]-c[0][1]
        yaw_rad = math.atan2(dy, dx)
        return float(yaw_rad)

    def image_callback(self, msg):
        if not self.camera_ready:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            if self.use_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is None or len(ids)==0:
                return

            ids = ids.flatten().astype(int)
            display = undistorted.copy()
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            corner_pixels = {}
            for i, marker_id in enumerate(ids):
                if marker_id in self.corner_ids:
                    c = corners[i][0]
                    cx, cy = np.mean(c[:,0]), np.mean(c[:,1])
                    corner_pixels[marker_id] = [cx, cy]
            if len(corner_pixels)<4:
                return

            self.pixel_matrix = [corner_pixels[i] for i in self.corner_ids]
            self.world_matrix = [self.world_coords[i] for i in self.corner_ids]
            self.H, _ = cv2.findHomography(np.array(self.pixel_matrix,np.float32),
                                           np.array(self.world_matrix,np.float32))
            if self.H is None:
                return

            bot_poses = {}
            crate_poses = {}
            for i, marker_id in enumerate(ids):
                if marker_id in self.corner_ids:
                    continue
                c = corners[i][0]
                cx, cy = np.mean(c[:,0]), np.mean(c[:,1])
                wx, wy = self.pixel_to_world(cx, cy)
                yaw = self.compute_yaw(corners[i])
                if marker_id in self.prev_yaw:
                    yaw = 0.7*self.prev_yaw[marker_id] + 0.3*yaw
                self.prev_yaw[marker_id] = yaw

                if marker_id in self.bot_ids:
                    bot_poses[int(marker_id)] = (wx, wy, yaw)
                else:
                    crate_poses[int(marker_id)] = (wx, wy, yaw)

                text = f"ID:{marker_id} X:{wx:.0f} Y:{wy:.0f} Yaw:{math.degrees(yaw):.1f}°"
                cv2.putText(display, text, (int(cx), int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if bot_poses:
                self.publish_poses(bot_poses,self.bot_pub)
            if crate_poses:
                self.publish_poses(crate_poses,self.crate_pub)

            cv2.imshow("Detected Markers", display)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {str(e)}")

    def publish_poses(self,poses,publisher):
        msg = Poses2D()
        for marker_id,(x,y,yaw) in poses.items():
            pose = Pose2D()
            pose.id = int(marker_id)
            pose.x = float(x)
            pose.y = float(y)
            pose.w = float(yaw)
            msg.poses.append(pose)
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HolonomicPoseDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()