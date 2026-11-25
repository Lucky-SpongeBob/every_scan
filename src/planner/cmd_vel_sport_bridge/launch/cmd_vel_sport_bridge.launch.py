from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cmd_vel_sport_bridge",
                executable="cmd_vel_sport_bridge",
                name="cmd_vel_sport_bridge",
                output="screen",
                parameters=[
                    {
                        "cmd_vel_topic": "/cmd_vel",
                        "sport_request_topic": "/api/sport/request",
                        "imu_yaw_topic": "/go2/imu_yaw",
                        "api_id": 1008,
                        "repeat_rate_hz": 100.0,
                        "command_hold_time": 0.25,
                        "command_timeout": 0.5,
                    }
                ],
            )
        ]
    )
