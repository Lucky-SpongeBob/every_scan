#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"

namespace cmd_vel_sport_bridge
{
class CmdVelSportBridge : public rclcpp::Node
{
public:
  CmdVelSportBridge() : Node("cmd_vel_sport_bridge")
  {
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    sport_request_topic_ = declare_parameter<std::string>(
        "sport_request_topic", "/api/sport/request");
    imu_yaw_topic_ = declare_parameter<std::string>(
        "imu_yaw_topic", "/go2/imu_yaw");
    api_id_ = declare_parameter<int>("api_id", 1008);
    command_timeout_ = declare_parameter<double>("command_timeout", 0.5);
    command_hold_time_ = declare_parameter<double>("command_hold_time", 0.25);
    repeat_rate_hz_ = declare_parameter<double>("repeat_rate_hz", 100.0);
    if (command_timeout_ <= 0.0)
      command_timeout_ = 0.5;
    if (command_hold_time_ <= 0.0)
      command_hold_time_ = 0.25;
    if (command_hold_time_ > command_timeout_)
      command_hold_time_ = command_timeout_;
    if (repeat_rate_hz_ < 20.0)
      repeat_rate_hz_ = 20.0;
    if (repeat_rate_hz_ > 200.0)
      repeat_rate_hz_ = 200.0;

    request_pub_ = create_publisher<unitree_api::msg::Request>(
        sport_request_topic_, rclcpp::QoS(10).reliable());
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        cmd_vel_topic_, rclcpp::QoS(20).reliable(),
        [this](geometry_msgs::msg::Twist::ConstSharedPtr msg) { forward(msg); });
    imu_yaw_pub_ = create_publisher<std_msgs::msg::Float32>(
        imu_yaw_topic_, rclcpp::QoS(10).reliable());
    sport_state_sub_ =
        create_subscription<unitree_go::msg::SportModeState>(
        "/sportmodestate", rclcpp::QoS(10).best_effort(),
        [this](unitree_go::msg::SportModeState::ConstSharedPtr msg) {
            std_msgs::msg::Float32 yaw;
            yaw.data = msg->imu_state.rpy[2];
            imu_yaw_pub_->publish(yaw);
        });
    watchdog_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / repeat_rate_hz_),
        std::bind(&CmdVelSportBridge::watchdogCallback, this));

    RCLCPP_INFO(
        get_logger(),
        "Direct bridge ready: %s -> %s (api_id=%d, repeat=%.1f Hz, "
        "hold=%.2f s, timeout=%.2f s)",
        cmd_vel_topic_.c_str(), sport_request_topic_.c_str(), api_id_,
        repeat_rate_hz_, command_hold_time_, command_timeout_);
  }

private:
  void publishCommand(double x, double y, double yaw_rate)
  {
    unitree_api::msg::Request request;
    request.header.identity.api_id = api_id_;

    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(9)
         << "{\"x\":" << x
         << ",\"y\":" << y
         << ",\"z\":" << yaw_rate
         << '}';
    request.parameter = json.str();
    request_pub_->publish(request);
  }

  void forward(const geometry_msgs::msg::Twist::ConstSharedPtr & msg)
  {
    last_command_time_ = now();
    has_command_ = true;
    timed_out_ = false;
    last_linear_x_ = msg->linear.x;
    last_linear_y_ = msg->linear.y;
    last_angular_z_ = msg->angular.z;
    publishCommand(last_linear_x_, last_linear_y_, last_angular_z_);
    if (
        std::abs(last_linear_x_) > 1e-6 ||
        std::abs(last_linear_y_) > 1e-6 ||
        std::abs(last_angular_z_) > 1e-6)
    {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 500,
          "Nonzero command: linear.x=%.9g linear.y=%.9g angular.z=%.9g",
          msg->linear.x, msg->linear.y, msg->angular.z);
    }
    if (!logged_first_command_) {
      logged_first_command_ = true;
      RCLCPP_INFO(
          get_logger(), "First command: linear.x=%.9g linear.y=%.9g angular.z=%.9g",
          msg->linear.x, msg->linear.y, msg->angular.z);
    }
  }

  void watchdogCallback()
  {
    if (!has_command_ || timed_out_)
      return;
    const double age = (now() - last_command_time_).seconds();
    if (age <= command_hold_time_) {
      if (
          std::abs(last_linear_x_) > 1e-6 ||
          std::abs(last_linear_y_) > 1e-6 ||
          std::abs(last_angular_z_) > 1e-6)
      {
        publishCommand(last_linear_x_, last_linear_y_, last_angular_z_);
      }
      return;
    }
    if (age < command_timeout_)
      return;

    publishCommand(0.0, 0.0, 0.0);
    timed_out_ = true;
    RCLCPP_WARN(
        get_logger(), "No /cmd_vel for %.2f s; published one zero command",
        command_timeout_);
  }

  std::string cmd_vel_topic_;
  std::string sport_request_topic_;
  std::string imu_yaw_topic_;
  int api_id_;
  double command_timeout_;
  double command_hold_time_;
  double repeat_rate_hz_;
  bool logged_first_command_{false};
  bool has_command_{false};
  bool timed_out_{false};
  double last_linear_x_{0.0};
  double last_linear_y_{0.0};
  double last_angular_z_{0.0};
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr imu_yaw_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr
    sport_state_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};
}  // namespace cmd_vel_sport_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cmd_vel_sport_bridge::CmdVelSportBridge>());
  rclcpp::shutdown();
  return 0;
}
