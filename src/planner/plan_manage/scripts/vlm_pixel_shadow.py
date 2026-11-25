#!/usr/bin/env python3
"""Ask a VLM for one floor pixel and visualize its metric ground projection."""

import argparse
from collections import deque
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from vlm_common import load_dotenv, query_json


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0
        * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        ),
        1.0
        - 2.0
        * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        ),
    )


def image_message_to_bgr(message):
    if message.encoding not in ("bgr8", "rgb8"):
        raise ValueError("unsupported image encoding %s" % message.encoding)
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )
    image = row[:, : message.width * 3].reshape(
        message.height, message.width, 3
    )
    if message.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def bgr_to_image_message(image, header):
    message = Image()
    message.header = header
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = np.ascontiguousarray(image).tobytes()
    return message


def quaternion_matrix(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def annotate_query_image(image):
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for index in range(1, 10):
        x = int(width * index / 10.0)
        y = int(height * index / 10.0)
        cv2.line(annotated, (x, 0), (x, height), (205, 205, 205), 1)
        cv2.line(annotated, (0, y), (width, y), (205, 205, 205), 1)
    for index in range(10):
        x = int((width - 1) * index / 10.0)
        y = int((height - 1) * index / 10.0)
        cv2.putText(
            annotated,
            "%.1f" % (index / 10.0),
            (max(2, min(width - 42, x + 4)), 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 230, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "%.1f" % (index / 10.0),
            (4, min(height - 4, y + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 230, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def stamp_key(stamp):
    return int(stamp.sec), int(stamp.nanosec)


def stamp_ns(stamp):
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


def format_stamp(stamp):
    milliseconds = int(stamp.nanosec) // 1000000
    return time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(int(stamp.sec))
    ) + ".%03d" % milliseconds


def annotate_temporal_image(image, label, stamp, age_s):
    annotated = annotate_query_image(image)
    text = "%s  %s  age=%.2fs" % (
        label,
        format_stamp(stamp),
        age_s,
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (min(annotated.shape[1] - 1, 760), 34),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        annotated,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 230, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


class VlmPixelShadow(Node):
    def __init__(self, args):
        super().__init__("vlm_pixel_shadow")
        self.args = args
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_image_header = None
        self.image_history = deque(maxlen=3)
        self.last_history_at = 0.0
        self.query_history = deque(maxlen=2)
        self.latest_camera_info = None
        self.latest_cloud = None
        self.latest_cloud_header = None
        self.latest_odom = None
        self.latest_imu_yaw = None
        self.last_imu_yaw_monotonic = 0.0
        self.motion_samples = deque(maxlen=40)
        self.motion_ready = False
        self.is_moving = False
        self.stable_since = 0.0
        self.last_odom_monotonic = 0.0
        self.busy = False
        self.last_query_at = 0.0
        self.locked_result = None
        self.goal_published = False
        self.pending_rotation = None
        self.pending_backup = None
        self.rotation_active = False
        self.rotation_start_yaw = 0.0
        self.rotation_requested_rad = 0.0
        self.rotation_target_yaw = None
        self.rotation_started_at = 0.0
        self.rotation_deadline = 0.0
        self.rotation_resume_at = 0.0
        self.backup_active = False
        self.backup_start_x = 0.0
        self.backup_start_y = 0.0
        self.backup_start_yaw = 0.0
        self.backup_target_distance = 0.0
        self.backup_travel = 0.0
        self.backup_deadline = 0.0
        self.auto_active = False
        self.auto_phase = "idle"
        self.auto_stop_requested = False
        self.auto_action_started_at = 0.0
        self.auto_action = "none"
        self.auto_cycle = 0
        self.auto_message = ""
        self.auto_last_publish = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        camera_info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Image, args.image_topic, self.image_callback, image_qos
        )
        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            camera_info_qos,
        )
        self.create_subscription(
            PointCloud2,
            args.ground_cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32,
            args.imu_yaw_topic,
            self.imu_yaw_callback,
            qos_profile_sensor_data,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, args.marker_topic, marker_qos
        )
        self.image_pub = self.create_publisher(
            Image, args.image_output_topic, image_qos
        )
        self.status_pub = self.create_publisher(
            String, args.status_topic, status_qos
        )
        self.auto_status_pub = self.create_publisher(
            String, "/vlm_pixel_shadow/auto_status", status_qos
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, args.cmd_vel_topic, 10
        )
        self.execution_pause_pub = self.create_publisher(
            Bool, "/planning/go2_execution_pause", 10
        )
        self.goal_pub = None
        self.approval_sub = None
        if args.drive:
            self.goal_pub = self.create_publisher(
                PoseStamped, args.goal_topic, 10
            )
            self.approval_sub = self.create_subscription(
                String,
                args.approval_topic,
                self.approval_callback,
                10,
            )
        self.create_timer(0.5, self.maybe_query)
        self.create_timer(0.1, self.rotation_tick)
        self.create_timer(0.05, self.backup_tick)
        self.create_timer(0.2, self.auto_tick)
        self.get_logger().info(
            "VLM pixel shadow ready; one-shot drive=%s"
            % args.drive
        )

    def image_callback(self, message):
        try:
            image = image_message_to_bgr(message)
        except Exception as exc:
            self.get_logger().warning("image rejected: %s" % exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_header = message.header
            now = time.monotonic()
            if (
                now - self.last_history_at
                >= self.args.history_interval
            ):
                self.image_history.append(
                    (message.header.stamp, image.copy())
                )
                self.last_history_at = now
            locked_result = self.locked_result
            camera_info = self.latest_camera_info
        if locked_result is not None and camera_info is not None:
            try:
                locked_result = self.reproject_locked_result(
                    locked_result, message.header, camera_info
                )
            except (
                TransformException,
                ValueError,
                cv2.error,
            ):
                locked_result = dict(locked_result)
                locked_result["projection_stale"] = True
                locked_result["pixel"] = None
                locked_result["visible"] = False
            self.publish_result(
                locked_result, image, message.header, republish=True
            )
        else:
            self.image_pub.publish(bgr_to_image_message(image, message.header))

    def camera_info_callback(self, message):
        with self.lock:
            self.latest_camera_info = message

    def cloud_callback(self, message):
        with self.lock:
            self.latest_cloud = message
            self.latest_cloud_header = message.header

    def odom_callback(self, message):
        now = time.monotonic()
        position = message.pose.pose.position
        sample = (
            now,
            float(position.x),
            float(position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        with self.lock:
            self.latest_odom = message
            if (
                self.last_odom_monotonic > 0.0
                and now - self.last_odom_monotonic > 0.5
            ):
                self.motion_samples.clear()
            self.last_odom_monotonic = now
            self.motion_samples.append(sample)
            while (
                self.motion_samples
                and now - self.motion_samples[0][0] > 1.0
            ):
                self.motion_samples.popleft()
            if len(self.motion_samples) < 2:
                self.motion_ready = False
                return
            oldest = self.motion_samples[0]
            span = now - oldest[0]
            if span < 0.7:
                self.motion_ready = False
                return
            distance = math.hypot(
                sample[1] - oldest[1],
                sample[2] - oldest[2],
            )
            yaw_change = abs(wrap_angle(sample[3] - oldest[3]))
            self.is_moving = (
                distance / span > 0.03
                or yaw_change / span > 0.05
            )
            self.motion_ready = True
            if self.is_moving:
                self.stable_since = 0.0
            elif self.stable_since <= 0.0:
                self.stable_since = now

    def imu_yaw_callback(self, message):
        with self.lock:
            self.latest_imu_yaw = float(message.data)
            self.last_imu_yaw_monotonic = time.monotonic()

    def rear_clearance(self):
        with self.lock:
            cloud = self.latest_cloud
            cloud_header = self.latest_cloud_header
        if cloud is None or cloud_header is None:
            return None
        frame = cloud_header.frame_id or self.args.distance_frame
        if frame not in ("body", "base", self.args.distance_frame):
            return None
        try:
            points = np.asarray(
                list(
                    point_cloud2.read_points(
                        cloud,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                    )
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            self.get_logger().warning(
                "cannot inspect rear LiDAR clearance: %s" % exc
            )
            return None
        if points.ndim != 2 or points.shape[0] < 20:
            return None
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ground_z = float(np.percentile(z, 5))
        cone = (
            (x < -0.45)
            & (x >= -self.args.rear_scan_range)
            & (np.abs(y) <= 0.16 + 0.12 * np.abs(x))
            & (z > ground_z + 0.15)
        )
        count = int(np.count_nonzero(cone))
        if count < 3:
            return {
                "distance_m": round(self.args.rear_scan_range, 3),
                "points": count,
                "frame": frame,
                "source": "clear",
            }
        distances = np.hypot(x[cone], y[cone])
        return {
            "distance_m": round(float(np.min(distances)), 3),
            "points": count,
            "frame": frame,
            "source": "lidar_rear_cone",
        }

    def current_yaw(self):
        now = time.monotonic()
        with self.lock:
            imu_yaw = self.latest_imu_yaw
            imu_age = now - self.last_imu_yaw_monotonic
            odom = self.latest_odom
            odom_age = now - self.last_odom_monotonic
        if imu_yaw is not None and imu_age <= 0.3:
            return wrap_angle(imu_yaw)
        if odom is not None and odom_age <= 0.5:
            return yaw_from_quaternion(
                odom.pose.pose.orientation
            )
        try:
            transform, _ = self.lookup_transform(
                self.args.distance_frame, Time()
            )
            return yaw_from_quaternion(transform.transform.rotation)
        except Exception as exc:
            self.get_logger().warning(
                "cannot read base yaw for approved rotation: %s" % exc
            )
            return None

    def publish_rotation_command(self, angular_z):
        command = Twist()
        command.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(command)

    def publish_execution_pause(self, paused):
        message = Bool()
        message.data = bool(paused)
        self.execution_pause_pub.publish(message)

    def start_rotation(self, angle_deg):
        if not self.args.drive:
            return False, "drive mode is disabled"
        with self.lock:
            if self.rotation_active or self.backup_active:
                return False, "another recovery motion is active"
        yaw = self.current_yaw()
        if yaw is None:
            return False, "base yaw is unavailable"
        try:
            requested = float(angle_deg)
        except (TypeError, ValueError):
            return False, "rotation angle is invalid"
        if not math.isfinite(requested) or abs(requested) < 1e-6:
            return False, "rotation angle is zero"
        requested = max(
            -self.args.max_rotate_deg,
            min(self.args.max_rotate_deg, requested),
        )
        if abs(requested) < self.args.min_rotate_deg:
            requested = math.copysign(
                self.args.min_rotate_deg, requested
            )
        control_requested = requested
        if requested > 0.0:
            control_requested += self.args.rotate_left_compensation_deg
        else:
            control_requested -= self.args.rotate_right_compensation_deg
        control_requested = max(
            -self.args.max_rotate_deg,
            min(self.args.max_rotate_deg, control_requested),
        )

        now = time.monotonic()
        self.publish_execution_pause(True)
        with self.lock:
            self.rotation_active = True
            self.rotation_start_yaw = yaw
            self.rotation_requested_rad = math.radians(
                control_requested
            )
            self.rotation_target_yaw = wrap_angle(
                yaw + math.radians(control_requested)
            )
            self.rotation_started_at = now
            self.rotation_deadline = (
                now
                + abs(math.radians(control_requested))
                / max(self.args.rotate_rate, 0.05)
                + 3.0
            )
            self.rotation_resume_at = 0.0
            self.pending_rotation = None
        self.publish_rotation_command(0.0)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "ok": True,
                        "walkable": False,
                        "action": "rotate",
                        "rotation_started": True,
                        "rotate_deg": round(requested, 2),
                        "approval_required": False,
                    },
                    ensure_ascii=False,
                )
            )
        )
        self.get_logger().info(
            "Approved rotation started: %+.1f deg" % requested
        )
        return True, None

    def rotation_tick(self):
        with self.lock:
            active = self.rotation_active
            target = self.rotation_target_yaw
            deadline = self.rotation_deadline
            start_yaw = self.rotation_start_yaw
            requested_rad = self.rotation_requested_rad
        if not active or target is None:
            return
        now = time.monotonic()
        yaw = self.current_yaw()
        if yaw is None:
            self.finish_rotation(False, "base yaw became unavailable")
            return

        error = wrap_angle(target - yaw)
        tolerance = math.radians(self.args.rotate_tolerance_deg)
        if abs(error) <= tolerance:
            actual = math.degrees(wrap_angle(yaw - start_yaw))
            self.finish_rotation(
                True,
                "rotation target reached at %.1f deg" % actual,
            )
            return
        if now >= deadline:
            actual_rad = wrap_angle(yaw - start_yaw)
            moved_enough = (
                actual_rad * requested_rad > 0.0
                and abs(actual_rad) >= 0.5 * abs(requested_rad)
            )
            self.finish_rotation(
                moved_enough,
                "rotation deadline at %.1f deg"
                % math.degrees(actual_rad),
            )
            return
        slow_window = math.radians(self.args.rotate_slow_window_deg)
        slow_phase = abs(error) <= slow_window
        rate_limit = (
            self.args.rotate_slow_rate
            if slow_phase
            else self.args.rotate_rate
        )
        command = max(
            -rate_limit,
            min(
                rate_limit,
                self.args.rotate_position_gain * error,
            ),
        )
        min_command = (
            self.args.rotate_slow_min_command
            if slow_phase
            else self.args.rotate_min_command
        )
        if abs(command) < min_command:
            command = math.copysign(
                min_command, error
            )
        self.publish_rotation_command(command)

    def finish_rotation(self, completed, reason):
        with self.lock:
            if not self.rotation_active:
                return
            self.rotation_active = False
            self.rotation_start_yaw = 0.0
            self.rotation_requested_rad = 0.0
            self.rotation_target_yaw = None
            self.rotation_resume_at = (
                time.monotonic() + self.args.rotation_settle
            )
            self.last_query_at = 0.0
            self.pending_rotation = None
        for _ in range(3):
            self.publish_rotation_command(0.0)
        self.publish_execution_pause(False)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "ok": True,
                        "walkable": False,
                        "action": "rotate_done"
                        if completed
                        else "rotate_failed",
                        "rotation_complete": bool(completed),
                        "rotation_reason": reason,
                        "approval_required": False,
                    },
                    ensure_ascii=False,
                )
            )
        )
        self.get_logger().info(
            "Approved rotation finished: %s; re-querying VLM" % reason
        )

    def publish_backup_command(self, linear_x, angular_z=0.0):
        command = Twist()
        command.linear.x = float(linear_x)
        command.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(command)

    def propose_backup(
        self, distance=None, source="manual", reason="", announce=True
    ):
        if not self.args.drive:
            return False, "drive mode is disabled"
        with self.lock:
            if self.backup_active:
                return False, "a backup is already active"
            if self.rotation_active:
                return False, "wait for the current rotation"
            odom = self.latest_odom
        if odom is None:
            return False, "odometry is unavailable"
        yaw = self.current_yaw()
        if yaw is None:
            return False, "base yaw is unavailable"
        rear = self.rear_clearance()
        if rear is None:
            return False, "rear LiDAR clearance is unavailable"
        rear_distance = float(rear["distance_m"])
        usable = max(
            0.0, rear_distance - self.args.backup_rear_margin
        )
        raw_requested = (
            self.args.backup_distance
            if distance is None
            else float(distance)
        )
        requested = min(raw_requested, usable)
        if requested < self.args.backup_min_distance:
            return False, "后面的空间太小，无法计算出有效后退距离"
        with self.lock:
            self.pending_backup = {
                "backup_distance_m": round(requested, 3),
                "backup_requested_m": round(raw_requested, 3),
                "backup_rear_clearance_m": round(rear_distance, 3),
                "backup_source": source,
                "reason": reason,
            }
            self.locked_result = None
            self.pending_rotation = None
            self.goal_published = False
            self.last_query_at = 0.0
        self.publish_rotation_command(0.0)
        self.publish_backup_command(0.0)
        if announce:
            self.status_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "ok": True,
                            "walkable": False,
                            "action": "backup",
                            "backup_approval_required": True,
                            "backup_distance_m": round(requested, 3),
                            "backup_requested_m": round(raw_requested, 3),
                            "backup_rear_clearance_m": round(
                                rear_distance, 3
                            ),
                            "backup_source": source,
                            "reason": reason,
                            "approval_required": True,
                        },
                        ensure_ascii=False,
                    )
                )
            )
        self.get_logger().info(
            "Backup proposal from %s: requested %.2f m, %.2f m available "
            "(rear clearance %.2f m); waiting for approval"
            % (source, raw_requested, requested, rear_distance)
        )
        return True, None

    def start_backup(self, distance=None, source="manual", reason=""):
        if not self.args.drive:
            return False, "drive mode is disabled"
        with self.lock:
            if self.backup_active:
                return False, "a backup is already active"
            if self.rotation_active:
                return False, "wait for the current rotation"
            odom = self.latest_odom
        if odom is None:
            return False, "odometry is unavailable"
        yaw = self.current_yaw()
        if yaw is None:
            return False, "base yaw is unavailable"
        rear = self.rear_clearance()
        if rear is None:
            return False, "rear LiDAR clearance is unavailable"
        rear_distance = float(rear["distance_m"])
        usable = rear_distance - self.args.backup_rear_margin
        raw_requested = (
            self.args.backup_distance
            if distance is None
            else float(distance)
        )
        requested = min(raw_requested, usable)
        if requested < self.args.backup_min_distance:
            return False, "not enough clear space behind the robot"

        self.publish_execution_pause(True)
        position = odom.pose.pose.position
        now = time.monotonic()
        with self.lock:
            self.backup_active = True
            self.backup_start_x = float(position.x)
            self.backup_start_y = float(position.y)
            self.backup_start_yaw = yaw
            self.backup_target_distance = requested
            self.backup_travel = 0.0
            self.backup_deadline = (
                now
                + requested / max(self.args.backup_speed, 0.03)
                + 3.0
            )
            self.locked_result = None
            self.pending_rotation = None
            self.pending_backup = None
            self.goal_published = False
            self.last_query_at = 0.0
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        self.marker_pub.publish(markers)
        self.publish_rotation_command(0.0)
        self.publish_backup_command(0.0)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "ok": True,
                        "walkable": False,
                        "action": "backup",
                        "backup_started": True,
                        "backup_distance_m": round(requested, 3),
                        "backup_requested_m": round(raw_requested, 3),
                        "backup_rear_clearance_m": round(
                            rear_distance, 3
                        ),
                        "backup_source": source,
                        "reason": reason,
                        "approval_required": False,
                    },
                    ensure_ascii=False,
                )
            )
        )
        self.get_logger().info(
            "Recovery backup started from %s: requested %.2f m, executing "
            "%.2f m (rear clearance %.2f m)"
            % (source, raw_requested, requested, rear_distance)
        )
        return True, None

    def backup_tick(self):
        with self.lock:
            active = self.backup_active
            start_x = self.backup_start_x
            start_y = self.backup_start_y
            start_yaw = self.backup_start_yaw
            target_distance = self.backup_target_distance
            deadline = self.backup_deadline
            odom = self.latest_odom
        if not active:
            return
        if odom is None:
            self.finish_backup(False, "odometry unavailable")
            return
        yaw = self.current_yaw()
        if yaw is None:
            self.finish_backup(False, "base yaw unavailable")
            return
        position = odom.pose.pose.position
        travel = (
            (float(position.x) - start_x) * math.cos(start_yaw)
            + (float(position.y) - start_y) * math.sin(start_yaw)
        )
        travel = max(0.0, -travel)
        with self.lock:
            self.backup_travel = travel
        stop_lead = min(
            self.args.backup_stop_lead,
            0.5 * target_distance,
        )
        if travel >= max(0.0, target_distance - stop_lead):
            self.finish_backup(True, "target distance reached")
            return
        if time.monotonic() >= deadline:
            self.finish_backup(
                False,
                "backup timeout after %.3f m" % travel,
            )
            return
        rear = self.rear_clearance()
        if (
            rear is not None
            and float(rear["distance_m"])
            <= self.args.backup_rear_margin
        ):
            self.finish_backup(False, "rear obstacle too close")
            return
        yaw_error = wrap_angle(start_yaw - yaw)
        yaw_command = 0.0
        if abs(yaw_error) > math.radians(2.0):
            yaw_command = max(-0.08, min(0.08, 1.2 * yaw_error))
        self.publish_backup_command(
            -self.args.backup_speed, yaw_command
        )

    def finish_backup(self, completed, reason):
        with self.lock:
            if not self.backup_active:
                return
            traveled = self.backup_travel
            self.backup_active = False
            self.backup_target_distance = 0.0
            self.backup_travel = 0.0
            self.backup_deadline = 0.0
            self.rotation_resume_at = (
                time.monotonic() + self.args.rotation_settle
            )
            self.last_query_at = 0.0
        for _ in range(3):
            self.publish_backup_command(0.0)
        self.publish_execution_pause(False)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "ok": True,
                        "walkable": False,
                        "action": "backup_done"
                        if completed
                        else "backup_failed",
                        "backup_complete": bool(completed),
                        "backup_result": reason,
                        "backup_travel_m": round(traveled, 3),
                        "approval_required": False,
                    },
                    ensure_ascii=False,
                )
            )
        )
        if completed:
            self.get_logger().info(
                "Manual recovery backup complete: %.3f m; re-querying VLM"
                % traveled
            )
        else:
            self.get_logger().warning(
                "Manual recovery backup stopped after %.3f m: %s"
                % (traveled, reason)
            )

    def motion_state(self):
        now = time.monotonic()
        with self.lock:
            if (
                self.last_odom_monotonic <= 0.0
                or now - self.last_odom_monotonic > 0.5
            ):
                return "unknown", 0.0
            if not self.motion_ready:
                return "settling", 0.0
            if self.is_moving:
                return "moving", 0.0
            stable_for = (
                0.0
                if self.stable_since <= 0.0
                else now - self.stable_since
            )
            return "stationary", stable_for

    def publish_auto_status(self, force=False):
        motion, stable_for = self.motion_state()
        now = time.monotonic()
        with self.lock:
            if not force and now - self.auto_last_publish < 0.5:
                return
            self.auto_last_publish = now
            data = {
                "ok": True,
                "auto_active": self.auto_active,
                "auto_phase": self.auto_phase,
                "auto_stop_requested": self.auto_stop_requested,
                "auto_cycle": self.auto_cycle,
                "auto_action": self.auto_action,
                "auto_message": self.auto_message,
                "motion_state": motion,
                "motion_stable_s": round(stable_for, 2),
            }
        self.auto_status_pub.publish(
            String(data=json.dumps(data, ensure_ascii=False))
        )

    def clear_selection_state(self):
        with self.lock:
            had_candidate = self.locked_result is not None
            self.locked_result = None
            self.pending_rotation = None
            self.pending_backup = None
            self.goal_published = False
            self.last_query_at = 0.0
            self.rotation_active = False
            self.rotation_start_yaw = 0.0
            self.rotation_requested_rad = 0.0
            self.rotation_target_yaw = None
            self.backup_active = False
        self.publish_rotation_command(0.0)
        self.publish_backup_command(0.0)
        self.publish_execution_pause(False)
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        self.marker_pub.publish(markers)
        return had_candidate

    def start_auto_mode(self):
        if not self.args.drive:
            self.get_logger().warning(
                "Auto execution ignored: drive mode is disabled"
            )
            return
        with self.lock:
            if self.auto_active:
                self.auto_message = "自动执行已在运行"
            else:
                self.auto_active = True
                self.auto_phase = "waiting_still"
                self.auto_stop_requested = False
                self.auto_action_started_at = 0.0
                self.auto_action = "none"
                self.auto_cycle = 0
                self.auto_message = "自动执行已启动，等待机械狗停稳"
        self.publish_auto_status(force=True)
        self.get_logger().info("Auto execution started")

    def request_auto_stop(self):
        now = time.monotonic()
        stop_now = False
        stop_message = None
        with self.lock:
            if not self.auto_active:
                self.auto_message = "自动执行未启动"
            elif self.auto_stop_requested:
                self.auto_message = "正在等待当前动作完成后结束"
            else:
                self.auto_stop_requested = True
                if self.auto_phase == "waiting_still":
                    stop_now = True
                    stop_message = "自动执行已结束"
                    self.auto_message = stop_message
                elif self.auto_phase == "selecting" and not self.busy:
                    stop_now = True
                    stop_message = "自动执行已结束（本次选点未执行）"
                    self.auto_message = stop_message
                else:
                    self.auto_message = (
                        "已请求结束，当前动作完成并停稳后停止"
                    )
        if stop_now:
            self.finish_auto(stop_message)
        else:
            self.publish_auto_status(force=True)
        self.get_logger().info(
            "Auto stop requested at %.2f s" % now
        )

    def start_auto_selection(self):
        had_candidate = self.clear_selection_state()
        with self.lock:
            self.auto_cycle += 1
            self.auto_phase = "selecting"
            self.auto_action = "none"
            self.auto_action_started_at = 0.0
            self.auto_message = (
                "第 %d 轮：正在让 VLM 重新选点" % self.auto_cycle
            )
            self.last_query_at = 0.0
        self.publish_auto_status(force=True)
        self.get_logger().info(
            "Auto cycle %d: requesting a new VLM target "
            "(previous candidate=%s)"
            % (self.auto_cycle, had_candidate)
        )

    def finish_auto(self, message, failed=False):
        with self.lock:
            self.auto_active = False
            self.auto_phase = "idle"
            self.auto_stop_requested = False
            self.auto_action_started_at = 0.0
            self.auto_action = "none"
            self.auto_message = message
        self.publish_auto_status(force=True)
        if failed:
            self.get_logger().warning("Auto execution stopped: %s" % message)
        else:
            self.get_logger().info("Auto execution finished: %s" % message)

    def auto_approve_result(self, result):
        with self.lock:
            if (
                not self.auto_active
                or self.auto_phase != "selecting"
                or self.auto_stop_requested
            ):
                return False
        action = str(result.get("action", "none")).strip().lower()
        self.approval_callback(String(data="approve"))
        with self.lock:
            started = (
                self.goal_published
                or self.rotation_active
                or self.backup_active
            )
            if started:
                self.auto_action = (
                    action
                    if action in ("rotate", "backup")
                    else "goal"
                )
                self.auto_action_started_at = time.monotonic()
                self.auto_phase = "waiting_motion"
                self.auto_message = {
                    "rotate": "自动执行：正在旋转",
                    "backup": "自动执行：正在后退",
                    "goal": "自动执行：目标已发送，等待机械狗移动",
                }[self.auto_action]
        self.publish_auto_status(force=True)
        return started

    def auto_tick(self):
        with self.lock:
            if not self.auto_active:
                return
            phase = self.auto_phase
            stop_requested = self.auto_stop_requested
            busy = self.busy
            action = self.auto_action
            started_at = self.auto_action_started_at
            rotation_active = self.rotation_active
            backup_active = self.backup_active
        motion, stable_for = self.motion_state()
        now = time.monotonic()

        if phase == "waiting_still":
            if stop_requested:
                self.finish_auto("自动执行已结束")
                return
            if (
                motion == "stationary"
                and stable_for >= 0.8
                and not busy
            ):
                self.start_auto_selection()
                return
        elif phase == "selecting":
            if stop_requested and not busy:
                self.finish_auto("自动执行已结束（本次选点未执行）")
                return
        elif phase == "waiting_motion":
            if motion == "moving":
                with self.lock:
                    self.auto_phase = "waiting_stop"
                    self.auto_message = (
                        "自动执行：动作进行中，等待机械狗停稳"
                    )
                self.publish_auto_status(force=True)
            else:
                action_finished = (
                    now - started_at >= 0.5
                    and (
                        (action == "rotate" and not rotation_active)
                        or (action == "backup" and not backup_active)
                    )
                )
                if action_finished:
                    if stop_requested:
                        self.finish_auto(
                            "当前动作已完成，自动执行结束"
                        )
                        return
                    with self.lock:
                        self.auto_phase = "waiting_still"
                        self.auto_message = (
                            "自动执行：动作完成，等待重新选点"
                        )
                    self.publish_auto_status(force=True)
                elif now - started_at > 20.0:
                    self.finish_auto(
                        "动作启动超时，自动执行已停止",
                        failed=True,
                    )
                    return
        elif phase == "waiting_stop":
            if motion == "stationary" and stable_for >= 1.2:
                if stop_requested:
                    self.finish_auto("当前动作已完成，自动执行结束")
                    return
                with self.lock:
                    self.auto_phase = "waiting_still"
                    self.auto_message = (
                        "自动执行：已停稳，准备下一轮选点"
                    )
                self.publish_auto_status(force=True)

        self.publish_auto_status()

    def maybe_query(self):
        with self.lock:
            if self.args.lock_target and self.locked_result is not None:
                return
            if self.pending_rotation is not None:
                return
            if self.pending_backup is not None:
                return
        now = time.monotonic()
        with self.lock:
            if (
                self.rotation_active
                or self.backup_active
                or now < self.rotation_resume_at
            ):
                return
        if now - self.last_query_at < self.args.interval:
            return
        with self.lock:
            if (
                self.busy
                or self.latest_image is None
                or self.latest_camera_info is None
                or self.latest_cloud is None
            ):
                return
            image = self.latest_image.copy()
            header = self.latest_image_header
            camera_info = self.latest_camera_info
            current_key = stamp_key(header.stamp)
            previous_queries = list(self.query_history)
            camera_history = [
                item
                for item in self.image_history
                if stamp_key(item[0]) != current_key
            ]
            history = list(previous_queries)
            used = {stamp_key(item[0]) for item in history}
            for item in reversed(camera_history):
                key = stamp_key(item[0])
                if key not in used and key != current_key:
                    history.append(item)
                    used.add(key)
                if len(history) >= 2:
                    break
            history.sort(key=lambda item: stamp_ns(item[0]))
            history = history[-2:]
            if len(history) < 2:
                return
            self.busy = True
            self.last_query_at = now
        threading.Thread(
            target=self.query_worker,
            args=(image, header, camera_info, history),
            name="vlm-pixel-query",
            daemon=True,
        ).start()

    def query_worker(self, image, header, camera_info, history):
        try:
            result = self.query(image, header, camera_info, history)
            if (
                self.args.lock_target
                and result.get("ok")
                and result.get("walkable")
                and result.get("target_xyz")
            ):
                result["locked"] = True
                with self.lock:
                    self.locked_result = result
                self.get_logger().info(
                    "VLM floor target locked at pixel (%d,%d); use the RViz "
                    "reselect button to select a new target"
                    % tuple(result["pixel"])
                )
                if self.goal_pub is not None:
                    result["approval_required"] = True
                    self.get_logger().info(
                        "Waiting for RViz approval before publishing the goal"
                    )
            elif (
                result.get("action") == "backup"
                and self.goal_pub is not None
            ):
                started, error = self.propose_backup(
                    distance=result.get("backup_requested_m"),
                    source="vlm",
                    reason=result.get("reason", ""),
                    announce=False,
                )
                if started:
                    with self.lock:
                        pending = dict(self.pending_backup)
                    result.update(
                        {
                            "approval_required": True,
                            "backup_approval_required": True,
                            "backup_distance_m": pending[
                                "backup_distance_m"
                            ],
                            "backup_rear_clearance_m": pending[
                                "backup_rear_clearance_m"
                            ],
                            "backup_source": "vlm",
                        }
                    )
                    self.get_logger().info(
                        "VLM requested backup: %.2f m available "
                        "(rear clearance %.2f m); waiting for approval"
                        % (
                            pending["backup_distance_m"],
                            pending["backup_rear_clearance_m"],
                        )
                    )
                else:
                    result.update(
                        {
                            "ok": False,
                            "error": error,
                            "approval_required": False,
                            "backup_started": False,
                        }
                    )
                    self.get_logger().warning(
                        "VLM backup request ignored: %s" % error
                    )
            elif (
                result.get("action") == "rotate"
                and self.goal_pub is not None
            ):
                with self.lock:
                    self.locked_result = None
                    self.pending_rotation = dict(result)
                result["approval_required"] = True
                self.get_logger().info(
                    "No candidate in current view; VLM suggests %+.1f deg. "
                    "Click approval to rotate." % result["rotate_deg"]
                )
            auto_approved = False
            with self.lock:
                auto_selecting = (
                    self.auto_active
                    and self.auto_phase == "selecting"
                )
                auto_stop_requested = self.auto_stop_requested
            if auto_selecting and result.get("approval_required"):
                if auto_stop_requested:
                    self.finish_auto(
                        "自动执行已结束（本次选点未执行）"
                    )
                else:
                    auto_approved = self.auto_approve_result(result)
                    if not auto_approved:
                        self.finish_auto(
                            "自动批准失败，自动执行已停止",
                            failed=True,
                        )
            elif auto_selecting and not result.get("ok"):
                self.finish_auto(
                    "VLM 选点失败，自动执行已停止",
                    failed=True,
                )
            self.publish_result(
                result, image, header, republish=auto_approved
            )
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "goal_publishing": False,
                "unix_time": time.time(),
            }
            self.status_pub.publish(
                String(data=json.dumps(result, ensure_ascii=False))
            )
            self.get_logger().warning("VLM pixel query failed: %s" % exc)
            with self.lock:
                auto_selecting = (
                    self.auto_active
                    and self.auto_phase == "selecting"
                )
            if auto_selecting:
                self.finish_auto(
                    "VLM 查询异常，自动执行已停止",
                    failed=True,
                )
        finally:
            with self.lock:
                self.query_history.append(
                    (header.stamp, image.copy())
                )
                self.busy = False

    def publish_goal(self, result):
        target = result["target_xyz"]
        goal = PoseStamped()
        goal.header.frame_id = self.args.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(target[0])
        goal.pose.position.y = float(target[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.goal_published = True
        result["goal_publishing"] = True
        result["goal_topic"] = self.args.goal_topic
        self.get_logger().info(
            "Published one-shot VLM goal to %s: [%.3f, %.3f, 0.000]"
            % (
                self.args.goal_topic,
                goal.pose.position.x,
                goal.pose.position.y,
            )
        )

    def approval_callback(self, message):
        command = message.data.strip().lower()
        if command == "auto_start":
            self.start_auto_mode()
            return
        if command == "auto_stop":
            self.request_auto_stop()
            return
        if command in ("rotate_left_30", "rotate_right_30"):
            angle = 30.0 if command == "rotate_left_30" else -30.0
            started, error = self.start_rotation(angle)
            if not started:
                self.get_logger().warning(
                    "Direct 30-degree rotation ignored: %s" % error
                )
                self.status_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "ok": False,
                                "walkable": False,
                                "action": "rotate",
                                "rotation_started": False,
                                "rotate_deg": angle,
                                "error": error,
                                "approval_required": False,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            else:
                self.get_logger().info(
                    "Direct %s 30-degree rotation started"
                    % ("left" if angle > 0.0 else "right")
                )
            return
        if command == "reselect":
            had_candidate = self.clear_selection_state()
            self.status_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "ok": True,
                            "approval_required": False,
                            "reselecting": True,
                            "had_candidate": had_candidate,
                        }
                    )
                )
            )
            self.get_logger().info(
                "RViz requested reselect; clearing the locked VLM target"
            )
            return

        if command in ("backup", "backup_03"):
            started, error = self.propose_backup()
            if not started:
                self.get_logger().warning(
                    "Backup proposal ignored: %s" % error
                )
                self.status_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "ok": False,
                                "walkable": False,
                                "action": "backup",
                                "backup_started": False,
                                "error": error,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            return

        if command != "approve":
            self.get_logger().warning(
                "Ignoring unknown VLM approval command: %s" % command
            )
            return

        with self.lock:
            pending_backup = (
                None
                if self.pending_backup is None
                else dict(self.pending_backup)
            )
            pending_rotation = (
                None
                if self.pending_rotation is None
                else dict(self.pending_rotation)
            )
        if pending_backup is not None:
            started, error = self.start_backup(
                pending_backup.get("backup_distance_m"),
                source=pending_backup.get("backup_source", "manual"),
                reason=pending_backup.get("reason", ""),
            )
            if not started:
                with self.lock:
                    self.pending_backup = None
                self.get_logger().warning(
                    "Approved backup ignored: %s" % error
                )
                self.status_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "ok": False,
                                "walkable": False,
                                "action": "backup",
                                "backup_started": False,
                                "error": error,
                                "approval_required": False,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            return
        if pending_rotation is not None:
            started, error = self.start_rotation(
                pending_rotation.get("rotate_deg", 0.0)
            )
            if started:
                self.get_logger().info(
                    "Approved VLM rotation request executed"
                )
            else:
                self.get_logger().warning(
                    "Approved rotation ignored: %s" % error
                )
                self.status_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "ok": False,
                                "walkable": False,
                                "action": "rotate",
                                "rotation_started": False,
                                "error": error,
                                "approval_required": True,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            return

        with self.lock:
            if self.locked_result is None:
                self.get_logger().warning(
                    "Approval ignored: no VLM candidate is locked"
                )
                return
            if self.goal_published:
                self.get_logger().warning(
                    "Approval ignored: this VLM candidate was already sent"
                )
                return
            self.locked_result = dict(self.locked_result)
            self.locked_result["approval_required"] = False
            self.locked_result["approved"] = True
            approved_result = dict(self.locked_result)
        self.publish_goal(approved_result)
        self.status_pub.publish(
            String(data=json.dumps(approved_result, ensure_ascii=False))
        )

    def query(self, image, header, camera_info, history):
        frames = list(history[-2:]) + [(header.stamp, image)]
        current_ns = stamp_ns(header.stamp)
        query_images = []
        frame_lines = []
        query_image = None
        for index, (stamp, frame_image) in enumerate(frames):
            remaining = len(frames) - 1 - index
            if remaining == 0:
                label = "CURRENT"
            elif remaining == 1:
                label = "PREVIOUS"
            else:
                label = "PREVIOUS-%d" % remaining
            age_s = max(
                0.0, (current_ns - stamp_ns(stamp)) / 1e9
            )
            if (
                self.args.vlm_image_width > 0
                and frame_image.shape[1] > self.args.vlm_image_width
            ):
                scale = (
                    self.args.vlm_image_width / frame_image.shape[1]
                )
                frame_image = cv2.resize(
                    frame_image,
                    (
                        self.args.vlm_image_width,
                        max(
                            1,
                            int(round(frame_image.shape[0] * scale)),
                        ),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            annotated = annotate_temporal_image(
                frame_image, label, stamp, age_s
            )
            if remaining == 0:
                query_image = annotated
            encoded = cv2.imencode(
                ".jpg",
                annotated,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self.args.vlm_jpeg_quality,
                ],
            )[1].tobytes()
            query_images.append(
                {
                    "label": "IMAGE %s timestamp=%s age=%.2fs"
                    % (label, format_stamp(stamp), age_s),
                    "jpeg": encoded,
                }
            )
            frame_lines.append(
                "%s: %s (%.2fs before current)"
                % (label, format_stamp(stamp), age_s)
            )

        query_image = (
            image if query_image is None else query_image
        )
        cv2.imwrite(
            self.args.query_image_path,
            query_image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        prompt = (
            "下面按时间顺序附上机器人前视相机的历史帧和当前帧：\n%s\n"
            "三张图用于理解机器人视野变化。请重点依据 CURRENT，"
            "并结合 PREVIOUS 和 PREVIOUS-2 判断画面中的道路、开口和"
            "遮挡变化。\n"
            % "\n".join(frame_lines)
        ) + (
            "你在看室内四足机器人的前视鱼眼相机。根据图像中的地面、墙脚、"
            "家具和障碍物，识别一条从画面近处连续延伸的可通行通道，并估计"
            "通道横向中心线。选点必须靠近通道中心，远离人物、家具、桌椅腿"
            "和两侧边界；不要选地平线、远端未知区域、墙面或门后。优先选择"
            "清晰可见的中近景地面点，目标距机器人约3到5.5米，不要刻意选最远点。"
            "先判断当前视角内是否存在可靠的候选地面点。如果存在候选点，"
            "action 返回 none，backup_required 返回 false，不要建议后退。"
            "如果没有任何可靠候选点，不要猜测坐标，再判断是否需要后退。"
            "当正前方被墙、门框、家具或其他障碍物近距离封死，原地旋转也"
            "无法获得可走方向，或当前空间不足以转身时，应建议后退；此时"
            "action 返回 backup，backup_required 返回 true，并根据前方阻挡"
            "程度给出 0.1 到 0.3 米的后退距离 backup_distance_m。只需依据"
            "前视图像估计需要后退多少才能恢复通行空间，机器人后方净空由雷达"
            "再次限制。否则建议旋转：action=rotate，backup_required 返回"
            "false，direction 只能为 left 或 right，rotate_deg 填写旋转角度"
            "的正数幅值（0到90度）。应结合三帧图像，优先选择能看到道路、"
            "走廊或开阔地面的方向旋转，尽量一次转到有路可走的方向；"
            "不要向仍然被墙或近处障碍挡住的方向旋转。"
            "图中的灰色网格是归一化坐标，"
            "x向右、y向下，均从0到1。有候选点时返回该地面点的归一化坐标，"
            "不要返回世界坐标或距离。严格只输出JSON："
            '{"walkable":true,"x":0.52,"y":0.76,'
            '"action":"none","backup_required":false,'
            '"confidence":0.8,"reason":"20字内中文原因"}'
            "无候选点时："
            '{"walkable":false,"action":"rotate","backup_required":false,'
            '"direction":"left","rotate_deg":30,'
            '"confidence":0.8,"reason":"正前方无可用地面通道"}'
            "需要后退时："
            '{"walkable":false,"action":"backup","backup_required":true,'
            '"backup_distance_m":0.3,"confidence":0.8,'
            '"reason":"前方通道封死，需要后退恢复通行空间"}'
        )
        self.get_logger().info(
            "VLM temporal input: " + " | ".join(frame_lines)
        )
        started = time.monotonic()
        decision = query_json(self.args, prompt, query_images)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        walkable_value = decision.get("walkable", False)
        if isinstance(walkable_value, str):
            walkable = walkable_value.strip().lower() in (
                "1",
                "true",
                "yes",
            )
        else:
            walkable = bool(walkable_value)
        action = str(decision.get("action", "none")).strip().lower()
        if action in ("turn", "look_around", "look-around"):
            action = "rotate"
        backup_value = decision.get("backup_required", False)
        if isinstance(backup_value, str):
            backup_required = backup_value.strip().lower() in (
                "1",
                "true",
                "yes",
            )
        else:
            backup_required = bool(backup_value)
        if action in ("backup", "reverse", "back") or backup_required:
            requested = decision.get(
                "backup_distance_m",
                decision.get("distance_m", self.args.backup_distance),
            )
            try:
                requested = float(requested)
            except (TypeError, ValueError):
                requested = self.args.backup_distance
            if not math.isfinite(requested):
                requested = self.args.backup_distance
            requested = max(
                self.args.backup_min_distance,
                min(self.args.backup_distance, requested),
            )
            return {
                "ok": True,
                "walkable": False,
                "action": "backup",
                "backup_required": True,
                "backup_requested_by_vlm": True,
                "backup_requested_m": round(requested, 3),
                "backup_distance_m": round(requested, 3),
                "goal_publishing": False,
                "confidence": max(
                    0.0,
                    min(1.0, float(decision.get("confidence", 0.0))),
                ),
                "reason": str(decision.get("reason", ""))[:80],
                "elapsed_ms": round(elapsed_ms, 1),
            }
        if action == "rotate" or not walkable:
            angle = decision.get(
                "rotate_deg",
                decision.get("rotation_deg", 30.0),
            )
            try:
                angle = float(angle)
            except (TypeError, ValueError):
                angle = 30.0
            if not math.isfinite(angle) or abs(angle) < 1e-6:
                angle = 30.0
            direction = str(
                decision.get("direction", "")
            ).strip().lower()
            if direction in ("left", "ccw", "counterclockwise"):
                angle = abs(angle)
            elif direction in ("right", "cw", "clockwise"):
                angle = -abs(angle)
            angle = max(
                -self.args.max_rotate_deg,
                min(self.args.max_rotate_deg, angle),
            )
            if abs(angle) < self.args.min_rotate_deg:
                angle = math.copysign(self.args.min_rotate_deg, angle)
            return {
                "ok": True,
                "walkable": False,
                "action": "rotate",
                "rotate_deg": round(angle, 2),
                "rotation_started": False,
                "goal_publishing": False,
                "reason": str(decision.get("reason", ""))[:80],
                "elapsed_ms": round(elapsed_ms, 1),
            }

        x_norm = float(decision["x"])
        y_norm = float(decision["y"])
        if x_norm > 1.0 or y_norm > 1.0:
            x_norm /= image.shape[1]
            y_norm /= image.shape[0]
        x_norm = min(1.0, max(0.0, x_norm))
        y_norm = min(1.0, max(0.0, y_norm))
        u = min(image.shape[1] - 1, max(0, int(round(x_norm * (image.shape[1] - 1)))))
        v = min(image.shape[0] - 1, max(0, int(round(y_norm * (image.shape[0] - 1)))))

        camera_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(camera_info.d, dtype=np.float64).reshape(-1, 1)
        normalized = cv2.fisheye.undistortPoints(
            np.asarray([[[float(u), float(v)]]], dtype=np.float64),
            camera_matrix,
            distortion,
        ).reshape(2)
        ray_camera = np.array(
            [normalized[0], normalized[1], 1.0], dtype=np.float64
        )
        ray_camera /= np.linalg.norm(ray_camera)

        camera_frame = header.frame_id or camera_info.header.frame_id
        query_time = Time.from_msg(header.stamp)
        transform, tf_latest = self.lookup_transform(camera_frame, query_time)
        rotation = quaternion_matrix(transform.transform.rotation)
        origin = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        ray_world = rotation @ ray_camera
        ray_world /= np.linalg.norm(ray_world)

        ground_normal, ground_offset, ground_debug = self.ground_plane()
        denominator = float(np.dot(ground_normal, ray_world))
        if denominator >= -1e-5:
            raise RuntimeError("selected camera ray does not hit the ground")
        ray_length = -(
            float(np.dot(ground_normal, origin)) + ground_offset
        ) / denominator
        if ray_length < self.args.min_target_range:
            raise RuntimeError(
                "selected ground point is too close: %.3f m" % ray_length
            )
        if ray_length > self.args.max_range:
            raise RuntimeError(
                "ground intersection outside range: %.3f m" % ray_length
            )
        ground_intersection = origin + ray_length * ray_world
        target, target_debug = self.lidar_target_for_ray(
            origin,
            ray_world,
            ground_normal,
            ground_offset,
            ground_intersection,
            camera_frame,
            query_time,
            camera_info,
            np.asarray([u, v], dtype=np.float64),
        )
        camera_horizontal_range = math.hypot(
            float(target[0] - origin[0]),
            float(target[1] - origin[1]),
        )
        camera_slant_range = float(np.linalg.norm(target - origin))
        distance_transform, distance_tf_latest = self.lookup_transform(
            self.args.distance_frame, query_time
        )
        robot_origin = np.array(
            [
                distance_transform.transform.translation.x,
                distance_transform.transform.translation.y,
                distance_transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        robot_horizontal_range = math.hypot(
            float(target[0] - robot_origin[0]),
            float(target[1] - robot_origin[1]),
        )
        robot_slant_range = float(np.linalg.norm(target - robot_origin))
        if robot_horizontal_range > self.args.max_target_distance:
            raise RuntimeError(
                "selected target is too far from dog center: %.3f m > %.3f m"
                % (
                    robot_horizontal_range,
                    self.args.max_target_distance,
                )
            )
        return {
            "ok": True,
            "walkable": True,
            "action": "none",
            "goal_publishing": False,
            "pixel": [u, v],
            "pixel_normalized": [round(x_norm, 5), round(y_norm, 5)],
            "target_xyz": [round(float(value), 4) for value in target],
            "camera_origin": [round(float(value), 4) for value in origin],
            "robot_origin": [
                round(float(value), 4) for value in robot_origin
            ],
            "range_origin": self.args.distance_frame,
            "ray_camera": [round(float(value), 6) for value in ray_camera],
            "ray_world": [round(float(value), 6) for value in ray_world],
            "target_distance_m": round(robot_horizontal_range, 3),
            "target_slant_range_m": round(robot_slant_range, 3),
            "camera_target_distance_m": round(camera_horizontal_range, 3),
            "camera_target_slant_range_m": round(camera_slant_range, 3),
            "target_source": target_debug["source"],
            "ground_intersection_xyz": [
                round(float(value), 4) for value in ground_intersection
            ],
            "lidar_target": target_debug,
            "ground_plane": ground_debug,
            "locked": False,
            "confidence": max(
                0.0, min(1.0, float(decision.get("confidence", 0.0)))
            ),
            "reason": str(decision.get("reason", ""))[:80],
            "tf_latest": tf_latest,
            "distance_tf_latest": distance_tf_latest,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def lookup_transform(self, source_frame, stamp):
        tf_latest = False
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.frame_id,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.3),
            )
        except TransformException:
            transform = self.tf_buffer.lookup_transform(
                self.args.frame_id,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
            tf_latest = True
        return transform, tf_latest

    def lidar_target_for_ray(
        self,
        origin,
        ray_world,
        ground_normal,
        ground_offset,
        ground_intersection,
        camera_frame,
        query_time,
        camera_info,
        selected_pixel,
    ):
        with self.lock:
            cloud = self.latest_cloud
            cloud_header = self.latest_cloud_header
        if cloud is None or cloud_header is None:
            raise RuntimeError("no LiDAR cloud available for target selection")

        points = np.asarray(
            list(
                point_cloud2.read_points(
                    cloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ),
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[0] < 50:
            raise RuntimeError("LiDAR cloud is too sparse for target selection")

        transform, tf_latest = self.lookup_transform(
            cloud_header.frame_id, Time.from_msg(cloud_header.stamp)
        )
        rotation = quaternion_matrix(transform.transform.rotation)
        cloud_origin = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        points_world = points @ rotation.T + cloud_origin
        floor_error = np.abs(
            points_world @ ground_normal + ground_offset
        )
        floor_indices = np.flatnonzero(floor_error <= 0.15)

        try:
            world_to_camera = self.tf_buffer.lookup_transform(
                camera_frame,
                self.args.frame_id,
                query_time,
                timeout=Duration(seconds=0.3),
            )
            world_to_camera_latest = False
        except TransformException:
            world_to_camera = self.tf_buffer.lookup_transform(
                camera_frame,
                self.args.frame_id,
                Time(),
                timeout=Duration(seconds=0.5),
            )
            world_to_camera_latest = True
        camera_rotation = quaternion_matrix(
            world_to_camera.transform.rotation
        )
        camera_translation = np.array(
            [
                world_to_camera.transform.translation.x,
                world_to_camera.transform.translation.y,
                world_to_camera.transform.translation.z,
            ],
            dtype=np.float64,
        )
        points_camera = (
            points_world @ camera_rotation.T + camera_translation
        )
        camera_matrix = np.asarray(
            camera_info.k, dtype=np.float64
        ).reshape(3, 3)
        distortion = np.asarray(
            camera_info.d, dtype=np.float64
        ).reshape(-1, 1)
        pixel_error = np.full(len(points_world), np.inf, dtype=np.float64)
        forward = points_camera[:, 2] > 0.05
        if np.any(forward):
            projected, _ = cv2.fisheye.projectPoints(
                points_camera[forward].reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                camera_matrix,
                distortion,
            )
            projected = projected.reshape(-1, 2)
            finite = np.all(np.isfinite(projected), axis=1)
            finite_indices = np.flatnonzero(forward)[finite]
            pixel_error[finite_indices] = np.linalg.norm(
                projected[finite] - selected_pixel[np.newaxis, :],
                axis=1,
            )

        support_distance = None
        matched_floor_indices = floor_indices[
            pixel_error[floor_indices] <= 20.0
        ]
        if matched_floor_indices.size:
            support_distance = float(
                np.min(
                    np.linalg.norm(
                        points_world[matched_floor_indices]
                        - ground_intersection,
                        axis=1,
                    )
                )
            )
        ground_horizontal_range = math.hypot(
            float(ground_intersection[0] - origin[0]),
            float(ground_intersection[1] - origin[1]),
        )
        support_limit = max(0.35, 0.20 * ground_horizontal_range)
        supported = (
            support_distance is not None and support_distance <= support_limit
        )
        debug = {
            "source": (
                "camera_ground_intersection"
                if supported
                else "camera_ground_intersection_sparse_lidar"
            ),
            "ground_support_distance_m": (
                None
                if support_distance is None
                else round(support_distance, 3)
            ),
            "ground_support_limit_m": round(support_limit, 3),
            "ground_support_available": supported,
            "support_points": int(matched_floor_indices.size),
            "support_pixel_error_px": (
                None
                if not matched_floor_indices.size
                else round(
                    float(np.min(pixel_error[matched_floor_indices])), 3
                )
            ),
            "cloud_points": int(len(points)),
            "cloud_age_s": round(
                max(
                    0.0,
                    self.get_clock().now().nanoseconds * 1e-9
                    - Time.from_msg(cloud_header.stamp).nanoseconds * 1e-9,
                ),
                3,
            ),
            "tf_latest": tf_latest,
            "camera_tf_latest": world_to_camera_latest,
        }
        return ground_intersection, debug

    def reproject_locked_result(self, result, header, camera_info):
        target_world = np.asarray(result["target_xyz"], dtype=np.float64)
        camera_frame = header.frame_id or camera_info.header.frame_id
        query_time = Time.from_msg(header.stamp)
        tf_latest = False
        try:
            transform = self.tf_buffer.lookup_transform(
                camera_frame,
                self.args.frame_id,
                query_time,
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            transform = self.tf_buffer.lookup_transform(
                camera_frame,
                self.args.frame_id,
                Time(),
                timeout=Duration(seconds=0.15),
            )
            tf_latest = True

        rotation = quaternion_matrix(transform.transform.rotation)
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        target_camera = rotation @ target_world + translation
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.args.frame_id,
                self.args.distance_frame,
                query_time,
                timeout=Duration(seconds=0.05),
            )
            robot_tf_latest = False
        except TransformException:
            robot_transform = self.tf_buffer.lookup_transform(
                self.args.frame_id,
                self.args.distance_frame,
                Time(),
                timeout=Duration(seconds=0.15),
            )
            robot_tf_latest = True
        robot_origin = np.array(
            [
                robot_transform.transform.translation.x,
                robot_transform.transform.translation.y,
                robot_transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        target_from_robot = target_world - robot_origin
        target_from_camera = target_camera
        projected = dict(result)
        projected["reprojected"] = True
        projected["reprojection_tf_latest"] = tf_latest
        projected["robot_origin"] = [
            round(float(value), 4) for value in robot_origin
        ]
        projected["range_origin"] = self.args.distance_frame
        projected["target_distance_m"] = round(
            float(np.linalg.norm(target_from_robot[:2])), 3
        )
        projected["target_slant_range_m"] = round(
            float(np.linalg.norm(target_from_robot)), 3
        )
        projected["camera_target_distance_m"] = round(
            float(np.linalg.norm(target_from_camera[:2])), 3
        )
        projected["camera_target_slant_range_m"] = round(
            float(np.linalg.norm(target_from_camera)), 3
        )
        projected["distance_tf_latest"] = robot_tf_latest
        projected["visible"] = False
        if target_camera[2] <= 0.05:
            projected["pixel"] = None
            return projected

        camera_matrix = np.asarray(
            camera_info.k, dtype=np.float64
        ).reshape(3, 3)
        distortion = np.asarray(
            camera_info.d, dtype=np.float64
        ).reshape(-1, 1)
        image_point, _ = cv2.fisheye.projectPoints(
            target_camera.reshape(1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            distortion,
        )
        u, v = image_point.reshape(2)
        if not np.isfinite(u) or not np.isfinite(v):
            projected["pixel"] = None
            return projected
        u_int = int(round(float(u)))
        v_int = int(round(float(v)))
        projected["pixel"] = [u_int, v_int]
        projected["visible"] = (
            0 <= u_int < camera_info.width
            and 0 <= v_int < camera_info.height
        )
        return projected

    def ground_plane(self):
        if self.args.ground_z is not None:
            return (
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                -float(self.args.ground_z),
                {
                    "source": "argument",
                    "z": round(float(self.args.ground_z), 4),
                    "normal": [0.0, 0.0, 1.0],
                },
            )

        with self.lock:
            cloud = self.latest_cloud
            cloud_header = self.latest_cloud_header
        if cloud is None or cloud_header is None:
            raise RuntimeError("no LiDAR cloud available for ground estimation")
        try:
            points = np.asarray(
                list(
                    point_cloud2.read_points(
                        cloud,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                    )
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            raise RuntimeError("cannot read LiDAR cloud: %s" % exc)
        if points.ndim != 2 or points.shape[0] < 50:
            raise RuntimeError("LiDAR cloud is too sparse for ground estimation")

        transform, tf_latest = self.lookup_transform(
            cloud_header.frame_id, Time.from_msg(cloud_header.stamp)
        )
        rotation = quaternion_matrix(transform.transform.rotation)
        origin = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        points_world = points @ rotation.T + origin
        radius = np.hypot(
            points_world[:, 0] - origin[0],
            points_world[:, 1] - origin[1],
        )
        ground_candidates = points_world[
            (radius > 0.45)
            & (radius < 6.0)
            & (points_world[:, 2] < origin[2] - 0.15)
        ]
        if len(ground_candidates) < 30:
            raise RuntimeError("not enough nearby points to estimate the ground")

        z_values = ground_candidates[:, 2]
        lower = min(float(np.percentile(z_values, 1)) - 0.04, origin[2] - 1.2)
        upper = float(origin[2] - 0.15)
        bins = np.arange(lower, upper + 0.0201, 0.02)
        counts, edges = np.histogram(z_values, bins=bins)
        if len(counts) < 9:
            raise RuntimeError("LiDAR ground histogram is too short")
        smoothed = np.convolve(counts, np.ones(5, dtype=np.int64), mode="same")
        minimum = max(25, int(0.40 * float(np.max(smoothed))))
        peak_indices = [
            index
            for index in range(2, len(smoothed) - 2)
            if smoothed[index] >= minimum
            and smoothed[index] >= smoothed[index - 2]
            and smoothed[index] >= smoothed[index + 2]
        ]
        if not peak_indices:
            raise RuntimeError("LiDAR ground histogram has no stable peak")
        groups = [[peak_indices[0]]]
        for index in peak_indices[1:]:
            if float(edges[index] - edges[groups[-1][-1]]) <= 0.10:
                groups[-1].append(index)
            else:
                groups.append([index])
        floor_group = min(
            groups, key=lambda group: float(np.mean([edges[index] for index in group]))
        )
        group_peak = max(floor_group, key=lambda index: int(smoothed[index]))
        run_low = float(edges[group_peak]) - 0.045
        run_high = float(edges[group_peak]) + 0.045
        inliers = ground_candidates[
            (ground_candidates[:, 2] >= run_low)
            & (ground_candidates[:, 2] <= run_high)
        ]
        required_inliers = max(12, int(0.015 * len(points)))
        if len(inliers) < required_inliers:
            run_low = float(edges[group_peak]) - 0.08
            run_high = float(edges[group_peak]) + 0.08
            inliers = ground_candidates[
                (ground_candidates[:, 2] >= run_low)
                & (ground_candidates[:, 2] <= run_high)
            ]
        if len(inliers) < required_inliers:
            raise RuntimeError("LiDAR ground fit has too few inliers")

        design = np.column_stack(
            (inliers[:, 0], inliers[:, 1], np.ones(len(inliers)))
        )
        slope_x, slope_y, intercept = np.linalg.lstsq(
            design, inliers[:, 2], rcond=None
        )[0]
        normal_raw = np.array(
            [-slope_x, -slope_y, 1.0], dtype=np.float64
        )
        normal_norm = float(np.linalg.norm(normal_raw))
        normal = normal_raw / normal_norm
        offset = -float(intercept) / normal_norm
        ground_z = float(
            np.median(
                slope_x * inliers[:, 0]
                + slope_y * inliers[:, 1]
                + intercept
            )
        )
        if abs(float(normal[2])) < 0.98:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            ground_z = float(np.median(inliers[:, 2]))
            offset = -ground_z
        return (
            normal,
            offset,
            {
                "source": "lidar",
                "z": round(ground_z, 4),
                "normal": [round(float(value), 5) for value in normal],
                "inliers": int(len(inliers)),
                "cloud_points": int(len(points)),
                "tf_latest": tf_latest,
            },
        )

    def publish_result(self, result, image, header, republish=False):
        if not republish:
            self.status_pub.publish(
                String(data=json.dumps(result, ensure_ascii=False))
            )
        if not result.get("ok"):
            return

        annotated = image.copy()
        if result.get("pixel"):
            u, v = result["pixel"]
            orange = (0, 165, 255)
            red = (35, 35, 255)
            halo = annotated.copy()
            cv2.circle(halo, (u, v), 34, orange, -1, cv2.LINE_AA)
            cv2.addWeighted(halo, 0.18, annotated, 0.82, 0.0, annotated)
            cv2.circle(annotated, (u, v), 30, (15, 15, 15), 8, cv2.LINE_AA)
            cv2.circle(annotated, (u, v), 30, orange, 4, cv2.LINE_AA)
            cv2.circle(annotated, (u, v), 17, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(annotated, (u, v), 6, red, -1, cv2.LINE_AA)
            cv2.line(
                annotated,
                (u - 42, v),
                (u - 20, v),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                annotated,
                (u + 20, v),
                (u + 42, v),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                annotated,
                (u, v - 42),
                (u, v - 20),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                annotated,
                (u, v + 20),
                (u, v + 42),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            distance = float(result.get("target_distance_m", 0.0))
            confidence = 100.0 * float(result.get("confidence", 0.0))
            label = "VLM FLOOR TARGET"
            detail = "px (%d,%d)  DOG %.2f m  %.0f%%" % (
                u,
                v,
                distance,
                confidence,
            )
            label_x = max(12, min(image.shape[1] - 360, u + 46))
            label_y = max(60, min(image.shape[0] - 76, v - 82))
            cv2.rectangle(
                annotated,
                (label_x, label_y),
                (label_x + 336, label_y + 62),
                (18, 18, 18),
                -1,
            )
            cv2.rectangle(
                annotated,
                (label_x, label_y),
                (label_x + 336, label_y + 62),
                orange,
                2,
            )
            cv2.circle(
                annotated,
                (label_x + 20, label_y + 20),
                6,
                red,
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label,
                (label_x + 34, label_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                detail,
                (label_x + 14, label_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                orange,
                2,
                cv2.LINE_AA,
            )
        else:
            distance = float(result.get("target_distance_m", 0.0))
            message = "WAYPOINT OUT OF VIEW  %.2f m" % distance
            if result.get("pixel") is None:
                message = "WAYPOINT BEHIND CAMERA  %.2f m" % distance
            if result.get("action") == "backup":
                message = "VLM: BACKUP %.2f m - APPROVE" % float(
                    result.get("backup_distance_m", 0.0)
                )
            elif result.get("action") == "rotate":
                angle = float(result.get("rotate_deg", 0.0))
                side = "LEFT" if angle >= 0.0 else "RIGHT"
                message = "VLM: %s %.0f deg - APPROVE TO ROTATE" % (
                    side,
                    abs(angle),
                )
            box_width = min(image.shape[1] - 24, 430)
            box_left = (image.shape[1] - box_width) // 2
            cv2.rectangle(
                annotated,
                (box_left, 18),
                (box_left + box_width, 74),
                (18, 18, 18),
                -1,
            )
            cv2.rectangle(
                annotated,
                (box_left, 18),
                (box_left + box_width, 74),
                (0, 165, 255),
                2,
            )
            cv2.putText(
                annotated,
                message,
                (box_left + 18, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        self.image_pub.publish(bgr_to_image_message(annotated, header))

        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.args.frame_id
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        if not result.get("walkable") or not result.get("target_xyz"):
            self.marker_pub.publish(markers)
            return

        target = result["target_xyz"]
        origin = result["camera_origin"]
        stamp = self.get_clock().now().to_msg()
        ray = Marker()
        ray.header.frame_id = self.args.frame_id
        ray.header.stamp = stamp
        ray.ns = "vlm_pixel_ray"
        ray.id = 1
        ray.type = Marker.LINE_LIST
        ray.action = Marker.ADD
        ray.pose.orientation.w = 1.0
        ray.scale.x = 0.025
        ray.color.r, ray.color.g, ray.color.b, ray.color.a = (
            1.0,
            0.75,
            0.05,
            0.95,
        )
        from geometry_msgs.msg import Point

        ray.points = [Point(x=origin[0], y=origin[1], z=origin[2]),
                      Point(x=target[0], y=target[1], z=target[2])]
        markers.markers.append(ray)

        base = Marker()
        base.header.frame_id = self.args.frame_id
        base.header.stamp = stamp
        base.ns = "vlm_pixel_target"
        base.id = 2
        base.type = Marker.CYLINDER
        base.action = Marker.ADD
        base.pose.position.x = target[0]
        base.pose.position.y = target[1]
        base.pose.position.z = target[2] + 0.015
        base.pose.orientation.w = 1.0
        base.scale.x = base.scale.y = 0.24
        base.scale.z = 0.03
        base.color.r, base.color.g, base.color.b, base.color.a = (
            1.0,
            0.05,
            0.03,
            0.95,
        )
        markers.markers.append(base)

        stem = Marker()
        stem.header.frame_id = self.args.frame_id
        stem.header.stamp = stamp
        stem.ns = "vlm_pixel_target"
        stem.id = 3
        stem.type = Marker.CYLINDER
        stem.action = Marker.ADD
        stem.pose.position.x = target[0]
        stem.pose.position.y = target[1]
        stem.pose.position.z = target[2] + 0.4
        stem.pose.orientation.w = 1.0
        stem.scale.x = stem.scale.y = 0.06
        stem.scale.z = 0.8
        stem.color.r, stem.color.g, stem.color.b, stem.color.a = (
            1.0,
            0.05,
            0.03,
            0.95,
        )
        markers.markers.append(stem)

        head = Marker()
        head.header.frame_id = self.args.frame_id
        head.header.stamp = stamp
        head.ns = "vlm_pixel_target"
        head.id = 4
        head.type = Marker.SPHERE
        head.action = Marker.ADD
        head.pose.position.x = target[0]
        head.pose.position.y = target[1]
        head.pose.position.z = target[2] + 0.86
        head.pose.orientation.w = 1.0
        head.scale.x = head.scale.y = head.scale.z = 0.22
        head.color.r, head.color.g, head.color.b, head.color.a = (
            1.0,
            0.05,
            0.03,
            0.95,
        )
        markers.markers.append(head)

        label = Marker()
        label.header.frame_id = self.args.frame_id
        label.header.stamp = stamp
        label.ns = "vlm_pixel_target"
        label.id = 5
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = target[0]
        label.pose.position.y = target[1]
        label.pose.position.z = target[2] + 1.08
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        label.color.r, label.color.g, label.color.b, label.color.a = (
            1.0,
            0.2,
            0.1,
            0.98,
        )
        if result.get("pixel"):
            label.text = "VLM (%d,%d) DOG %.2fm" % (
                result["pixel"][0],
                result["pixel"][1],
                result["target_distance_m"],
            )
        else:
            label.text = "VLM TARGET DOG %.2fm" % result[
                "target_distance_m"
            ]
        markers.markers.append(label)
        if not republish:
            self.marker_pub.publish(markers)
            self.get_logger().info(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-topic", default="/go2/front_camera/image_raw"
    )
    parser.add_argument(
        "--camera-info-topic", default="/go2/front_camera/camera_info"
    )
    parser.add_argument(
        "--marker-topic", default="/vlm_pixel_shadow"
    )
    parser.add_argument(
        "--ground-cloud-topic", default="/LIO/clouds_lidar"
    )
    parser.add_argument(
        "--distance-frame",
        default="base",
        help="robot-center frame used for the displayed target range",
    )
    parser.add_argument("--odom-topic", default="/LIO/odom_vehicle")
    parser.add_argument("--imu-yaw-topic", default="/go2/imu_yaw")
    parser.add_argument(
        "--rear-scan-range",
        type=float,
        default=3.0,
        help="maximum rear LiDAR range used for backup safety",
    )
    parser.add_argument(
        "--backup-distance",
        type=float,
        default=0.3,
        help="maximum manual backup distance in metres",
    )
    parser.add_argument(
        "--backup-speed",
        type=float,
        default=0.30,
        help="normalized Sport Move reverse speed",
    )
    parser.add_argument(
        "--backup-min-distance",
        type=float,
        default=0.02,
        help="minimum useful manual backup distance",
    )
    parser.add_argument(
        "--backup-rear-margin",
        type=float,
        default=0.02,
        help="clearance to preserve behind the robot",
    )
    parser.add_argument(
        "--backup-stop-lead",
        type=float,
        default=0.02,
        help="distance before the target at which reverse motion stops",
    )
    parser.add_argument(
        "--backup-max-linear-speed",
        type=float,
        default=0.10,
        help="maximum speed before a manual backup may start",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--rotate-rate",
        type=float,
        default=0.8,
        help="maximum normalized Sport Move yaw command",
    )
    parser.add_argument(
        "--rotate-min-command",
        type=float,
        default=0.50,
        help="minimum normalized yaw command that moves the robot",
    )
    parser.add_argument(
        "--rotate-position-gain",
        type=float,
        default=1.5,
        help="yaw command gain per radian of heading error",
    )
    parser.add_argument(
        "--rotate-tolerance-deg",
        type=float,
        default=3.0,
        help="yaw tolerance for an approved rotation",
    )
    parser.add_argument(
        "--rotate-slow-window-deg",
        type=float,
        default=6.0,
        help="remaining angle at which slow rotation begins",
    )
    parser.add_argument(
        "--rotate-slow-rate",
        type=float,
        default=0.50,
        help="normalized yaw command for the final slow phase",
    )
    parser.add_argument(
        "--rotate-slow-min-command",
        type=float,
        default=0.50,
        help="minimum normalized yaw command in the slow phase",
    )
    parser.add_argument(
        "--rotate-left-compensation-deg",
        type=float,
        default=-5.5,
        help="additional left rotation to offset settling drift",
    )
    parser.add_argument(
        "--rotate-right-compensation-deg",
        type=float,
        default=6.0,
        help="additional right rotation to offset settling drift",
    )
    parser.add_argument(
        "--min-rotate-deg",
        type=float,
        default=10.0,
        help="minimum approved rotation angle",
    )
    parser.add_argument(
        "--max-rotate-deg",
        type=float,
        default=90.0,
        help="maximum absolute approved rotation angle",
    )
    parser.add_argument(
        "--rotation-settle",
        type=float,
        default=1.0,
        help="seconds to wait after rotation before re-querying",
    )
    parser.add_argument(
        "--image-output-topic", default="/go2/front_camera/vlm_point_overlay"
    )
    parser.add_argument(
        "--status-topic", default="/vlm_pixel_shadow/status"
    )
    parser.add_argument("--frame-id", default="world")
    parser.add_argument(
        "--ground-z",
        type=float,
        default=None,
        help="override the LiDAR-estimated ground height in the world frame",
    )
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument(
        "--max-target-distance",
        type=float,
        default=5.5,
        help="maximum accepted horizontal target range from the distance frame",
    )
    parser.add_argument("--min-target-range", type=float, default=0.1)
    parser.add_argument("--interval", type=float, default=8.0)
    parser.add_argument(
        "--history-interval",
        type=float,
        default=1.0,
        help="seconds between cached VLM history frames",
    )
    parser.add_argument(
        "--vlm-image-width",
        type=int,
        default=480,
        help="maximum image width sent to the VLM",
    )
    parser.add_argument(
        "--vlm-jpeg-quality",
        type=int,
        default=65,
        help="JPEG quality for images sent to the VLM",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="publish one goal only after RViz approval of the locked target",
    )
    parser.add_argument(
        "--goal-topic", default="/move_base_simple/goal"
    )
    parser.add_argument(
        "--approval-topic", default="/go2/vlm_approval"
    )
    parser.add_argument(
        "--lock-target", dest="lock_target", action="store_true"
    )
    parser.add_argument(
        "--no-lock-target", dest="lock_target", action="store_false"
    )
    parser.set_defaults(lock_target=True, drive=False)
    parser.add_argument(
        "--query-image-path", default="/tmp/vlm_pixel_query.jpg"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8001/v1/chat/completions",
    )
    parser.add_argument("--model", default="unsloth/Qwen3.5-27B-GGUF")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="allow template-level reasoning in VLM responses",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--api-timeout", type=float, default=240.0)
    parser.add_argument(
        "--env-file", default=os.path.expanduser("~/.env")
    )
    args = parser.parse_args()
    if (
        args.interval <= 0.0
        or args.history_interval <= 0.0
        or args.vlm_image_width <= 0
        or args.vlm_jpeg_quality < 1
        or args.vlm_jpeg_quality > 100
        or args.max_range <= 0.1
        or args.max_target_distance <= args.min_target_range
        or args.min_target_range < 0.1
        or args.min_target_range >= args.max_range
        or args.rotate_rate <= 0.0
        or args.rotate_min_command <= 0.0
        or args.rotate_min_command > args.rotate_rate
        or args.rotate_position_gain <= 0.0
        or args.rotate_tolerance_deg <= 0.0
        or args.rotate_slow_window_deg <= 0.0
        or args.rotate_slow_rate <= 0.0
        or args.rotate_slow_rate > args.rotate_rate
        or args.rotate_slow_min_command <= 0.0
        or args.rotate_slow_min_command > args.rotate_slow_rate
        or abs(args.rotate_left_compensation_deg) > 15.0
        or args.rotate_right_compensation_deg < 0.0
        or args.rotate_right_compensation_deg > 15.0
        or args.min_rotate_deg <= 0.0
        or args.max_rotate_deg < args.min_rotate_deg
        or args.rotation_settle < 0.0
        or args.rear_scan_range <= 0.0
        or args.backup_distance <= 0.0
        or args.backup_speed <= 0.0
        or args.backup_min_distance <= 0.0
        or args.backup_rear_margin < 0.0
        or args.backup_stop_lead < 0.0
        or args.backup_max_linear_speed < 0.0
    ):
        parser.error("interval or target range settings are invalid")
    load_dotenv(args.env_file)
    rclpy.init()
    node = VlmPixelShadow(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
