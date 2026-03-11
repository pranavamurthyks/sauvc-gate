#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from collections import deque

from custom_msgs.msg import Telemetry, Commands
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float64MultiArray, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from custom_msgs.msg import Object2D


# Stages

# Stage 1 -> Identify bbox by yaw, allign laterally and surge forward
# Stage 2 -> When the gate is out of the frame, come back a bit to check if gate is visible to confirm whether it is loss of vision
#            of gate because of going near or some other stuff
# Stage 3 -> Move forward for 4 seconds and take u turn
# Stage 4 -> Return via the gate


class GateControlNode(Node):
    def __init__(self):
        super().__init__("gate_control_node")

        # PID Values
        self.kp_lateral = 100
        self.kp_surge = 100

        self.kd_lateral = 200
        self.kd_surge = 200


        # Variable Initialization
        self.stage = 0
        self.bbox_history = deque(maxlen=5)
        self.pose_history = deque(maxlen=5)
        self.z = None

        self.bbox_center_x = None
        self.bbox_center_y = None
        self.bbox_size_x = None
        self.bbox_size_y = None
        self.frame_center_x = None
        self.frame_center_y = None

        self.conf_threshold = 0.5

        # FIX 4 & 5: State variables for non-blocking stage timers
        self.stage_start_time = None
        self.stage3_phase = 0
        self.stage3_phase_start_time = None

        self.cmd = Commands()
        self.cmd.arm = False
        self.cmd.mode = "ALT_HOLD"
        self.cmd.forward = 1500
        self.cmd.lateral = 1500
        self.cmd.thrust = 1500
        self.cmd.yaw = 1500
        self.cmd.pitch = 1500
        self.cmd.roll = 1500

        self.qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)


        # Subscribers
        self.telemetry_sub = self.create_subscription(Telemetry, '/master/telemetry', self.telemetry_callback, self.qos)
        self.bbox_sub = self.create_subscription(Detection2DArray, "/vision/detections", self.bbox_callback, self.qos)
        self.frame_center_sub = self.create_subscription(Float64MultiArray, '/frame/center', self.frame_center_callback, self.qos)


        # Publishers
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)

        # Publish stable state
        self.cmd_pub.publish(self.cmd)


        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)


    # Callbacks
    def telemetry_callback(self, msg):
        self.cmd.arm = msg.arm
        self.yaw = msg.yaw



    def bbox_callback(self, msg):
        if not msg.detections:
            self.get_logger().debug("Empty detection array received")
            return
        
        for det in msg.detections:
            confidence = 0.0
            if det.results and len(det.results) > 0:
                confidence = det.results[0].hypothesis.score
                if confidence < self.conf_threshold:
                    self.get_logger().debug(f"Skipping detection (confidence {confidence:.2f} < threshold {self.conf_threshold})")
                    continue
            
            # FIX 7: Safe cast of det.id to int
            try:
                self.class_id = int(det.id)
            except (ValueError, TypeError):
                self.class_id = -1

            if (self.class_id == 4):
                self.bbox_history.append(
                {
                'cx': det.bbox.center.position.x,
                'cy': det.bbox.center.position.y,
                'w': det.bbox.size_x,
                'h': det.bbox.size_y
                })
        
                if (len(self.bbox_history) == 5): # better do this in percep and publish stage as per that 
                    self.bbox_center_x = np.mean([f['cx'] for f in self.bbox_history])
                    self.bbox_center_y = np.mean([f['cy'] for f in self.bbox_history])
                    self.bbox_size_x = np.mean([f['w'] for f in self.bbox_history])
                    self.bbox_size_y = np.mean([f['h'] for f in self.bbox_history])



    def frame_center_callback(self, msg):
        # FIX 1: Access .data field of Float64MultiArray
        self.frame_center_x = msg.data[0]
        self.frame_center_y = msg.data[1]




    # Control loop
    def control_loop(self):
        if (self.stage == 1):
            self.control_one()
        elif (self.stage == 2):
            self.control_two()
        elif (self.stage == 3):
            self.control_three()


    def control_one(self):
        # FIX 6: Guard against None values before use
        if self.bbox_center_x is None or self.frame_center_x is None:
            return

        self.cmd.arm = True
        self.lateral_error = self.bbox_center_x - self.frame_center_x
        # FIX 2: Corrected misplaced parenthesis in abs() and replaced self.w with self.bbox_size_x
        if (abs(self.lateral_error) < 0.05 * self.bbox_size_x):
            self.cmd.forward = 1550
        else:
            self.lateral_cmd = 1500 + (self.kp_lateral * self.lateral_error)
            self.cmd.lateral = self.pwm_clamp(self.lateral_cmd)

    
    def control_two(self):
        # FIX 4: Non-blocking timer using state machine instead of while True
        if self.stage_start_time is None:
            self.cmd.forward = 1550
            self.cmd_pub.publish(self.cmd)
            self.stage_start_time = self.get_clock().now()

        # FIX 3: self.get_clock() called with ()
        delta = (self.get_clock().now() - self.stage_start_time).nanoseconds / 1e9
        if delta >= 6:
            self.stage = 3
            self.stage_start_time = None
            self.cmd.forward = 1500
            self.cmd_pub.publish(self.cmd)
        

    def control_three(self):
        # FIX 4 & 5: Non-blocking two-phase timer using stage3_phase and separate start times
        if self.stage3_phase == 0:
            if self.stage3_phase_start_time is None:
                self.cmd.yaw = 1550
                self.cmd_pub.publish(self.cmd)
                self.stage3_phase_start_time = self.get_clock().now()

            # FIX 3: self.get_clock() called with ()
            delta = (self.get_clock().now() - self.stage3_phase_start_time).nanoseconds / 1e9
            if delta >= 3:
                self.cmd.yaw = 1500
                self.cmd.forward = 1550
                self.cmd_pub.publish(self.cmd)
                # FIX 5: Reset start time for the second phase independently
                self.stage3_phase_start_time = self.get_clock().now()
                self.stage3_phase = 1

        elif self.stage3_phase == 1:
            # FIX 3: self.get_clock() called with ()
            delta = (self.get_clock().now() - self.stage3_phase_start_time).nanoseconds / 1e9
            if delta >= 3:
                self.cmd.forward = 1500
                self.cmd_pub.publish(self.cmd)
                self.stage3_phase = 2
            

    def pwm_clamp(self, pwm):
        return min(1700, max(1100, pwm))