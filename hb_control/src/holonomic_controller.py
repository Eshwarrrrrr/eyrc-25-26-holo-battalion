#!/usr/bin/env python3

"""
Holonomic Robot Controller with PID Control and Inverse Kinematics
Task 1C - Square Trajectory Navigation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from hb_interfaces.msg import Pose2D
import math
import time

class PIDController:
    """PID Controller for position and orientation control"""
    
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        
    def compute(self, error, current_time):
        """Compute PID output"""
        if self.prev_time is None:
            self.prev_time = current_time
            dt = 0.01
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.output_limit:
            self.integral = max(-self.output_limit/self.ki if self.ki != 0 else 0, 
                              min(self.output_limit/self.ki if self.ki != 0 else 0, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limit:
            output = max(-self.output_limit, min(self.output_limit, output))
        
        # Update for next iteration
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None


class HolonomicController(Node):
    """Main controller node for holonomic robot navigation"""
    
    def __init__(self):
        super().__init__('holonomic_controller')
        
        # Robot parameters (3-wheel holonomic drive)
        self.wheel_radius = 0.05  # meters
        self.robot_radius = 0.165  # meters (distance from center to wheel)
        
        # Wheel angles (120 degrees apart, starting from front)
        self.wheel_angles = [
            math.radians(90),   # Wheel 1 (front)
            math.radians(210),  # Wheel 2 (back-left)
            math.radians(330)   # Wheel 3 (back-right)
        ]
        
        # Goal waypoints for square trajectory
        self.goals = [
            (820.0, 920.0),
            (820.0, 1520.0),
            (1620.0, 1520.0),
            (1620.0, 920.0),
            (820.0, 920.0)
        ]
        self.current_goal_index = 0
        self.goal_reached_threshold = 10.0  # 10 units as per requirements
        
        # Current robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.pose_received = False
        
        # PID controllers - TUNED VALUES
        self.pid_x = PIDController(kp=0.8, ki=0.01, kd=0.3, output_limit=200.0)
        self.pid_y = PIDController(kp=0.8, ki=0.01, kd=0.3, output_limit=200.0)
        self.pid_theta = PIDController(kp=1.5, ki=0.02, kd=0.4, output_limit=2.0)
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(
            Pose2D,
            '/bot_pose',
            self.pose_callback,
            10
        )
        
        # Control loop timer (50 Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        # Status tracking
        self.goals_reached = 0
        self.mission_complete = False
        
        self.get_logger().info('Holonomic Controller initialized')
        self.get_logger().info(f'Target waypoints: {self.goals}')
    
    def pose_callback(self, msg):
        """Callback for robot pose updates"""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_theta = msg.theta
        self.pose_received = True
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def inverse_kinematics(self, vx, vy, omega):
        """
        Inverse kinematics for 3-wheel holonomic drive
        
        Args:
            vx: Linear velocity in x direction (robot frame)
            vy: Linear velocity in y direction (robot frame)
            omega: Angular velocity (rotation)
            
        Returns:
            List of wheel velocities [wheel1, wheel2, wheel3]
        """
        wheel_velocities = []
        
        for angle in self.wheel_angles:
            # Velocity component perpendicular to wheel axis
            v_wheel = (-vx * math.sin(angle) + vy * math.cos(angle) + 
                      omega * self.robot_radius)
            
            # Convert to angular velocity
            angular_vel = v_wheel / self.wheel_radius
            wheel_velocities.append(angular_vel)
        
        return wheel_velocities
    
    def control_loop(self):
        """Main control loop"""
        if not self.pose_received or self.mission_complete:
            return
        
        # Get current goal
        if self.current_goal_index >= len(self.goals):
            if not self.mission_complete:
                self.mission_complete = True
                self.stop_robot()
                self.get_logger().info('Mission Complete!')
                self.get_logger().info(f'Goals reached: {self.goals_reached}/{len(self.goals)}')
            return
        
        goal_x, goal_y = self.goals[self.current_goal_index]
        
        # Calculate errors in world frame
        error_x_world = goal_x - self.current_x
        error_y_world = goal_y - self.current_y
        distance_to_goal = math.sqrt(error_x_world**2 + error_y_world**2)
        
        # Check if goal reached
        if distance_to_goal < self.goal_reached_threshold:
            self.get_logger().info(
                f'Goal {self.current_goal_index + 1} reached: ({goal_x}, {goal_y})'
            )
            self.goals_reached += 1
            self.current_goal_index += 1
            
            # Reset PID controllers for next goal
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_theta.reset()
            
            return
        
        # Transform errors to robot frame
        cos_theta = math.cos(self.current_theta)
        sin_theta = math.sin(self.current_theta)
        error_x_robot = error_x_world * cos_theta + error_y_world * sin_theta
        error_y_robot = -error_x_world * sin_theta + error_y_world * cos_theta
        
        # Calculate desired orientation (point towards goal)
        desired_theta = math.atan2(error_y_world, error_x_world)
        error_theta = self.normalize_angle(desired_theta - self.current_theta)
        
        # Get current time
        current_time = time.time()
        
        # Compute PID outputs
        vx = self.pid_x.compute(error_x_robot, current_time)
        vy = self.pid_y.compute(error_y_robot, current_time)
        omega = self.pid_theta.compute(error_theta, current_time)
        
        # Apply velocity limits
        max_linear_vel = 150.0
        vel_magnitude = math.sqrt(vx**2 + vy**2)
        if vel_magnitude > max_linear_vel:
            scale = max_linear_vel / vel_magnitude
            vx *= scale
            vy *= scale
        
        # Calculate wheel velocities using inverse kinematics
        wheel_vels = self.inverse_kinematics(vx, vy, omega)
        
        # Publish velocity commands
        self.publish_wheel_velocities(wheel_vels)
        
        # Log progress periodically
        if self.get_clock().now().nanoseconds % 1000000000 < 20000000:  # ~Every second
            self.get_logger().info(
                f'Goal {self.current_goal_index + 1}: '
                f'Distance={distance_to_goal:.2f}, '
                f'Angle_error={math.degrees(error_theta):.2f}°'
            )
    
    def publish_wheel_velocities(self, wheel_vels):
        """Publish wheel velocities as Twist message"""
        twist = Twist()
        twist.linear.x = wheel_vels[0]
        twist.linear.y = wheel_vels[1]
        twist.linear.z = wheel_vels[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
    
    def stop_robot(self):
        """Stop the robot"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info('Robot stopped')


def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Controller interrupted')
    finally:
        controller.stop_robot()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()