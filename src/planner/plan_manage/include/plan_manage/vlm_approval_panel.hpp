#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <QLabel>
#include <QPushButton>
#include <QTimer>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/string.hpp>

namespace scan_planner
{

class VlmApprovalPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit VlmApprovalPanel(QWidget * parent = nullptr);
  void onInitialize() override;

private Q_SLOTS:
  void approveGoal();
  void requestReselect();
  void backupDistance();
  void rotateLeft30();
  void rotateRight30();
  void startAutoExecution();
  void stopAutoExecution();
  void refreshStatus();

private:
  void publishCommand(const std::string & command);
  void updateMotionLabel();
  void refreshAutoStatus();
  void odomCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr & message);

  struct MotionSample
  {
    std::chrono::steady_clock::time_point time;
    double x;
    double y;
    double yaw;
  };

  QLabel * motion_label_{nullptr};
  QLabel * auto_label_{nullptr};
  QLabel * status_label_{nullptr};
  QPushButton * approve_button_{nullptr};
  QPushButton * reselect_button_{nullptr};
  QPushButton * backup_button_{nullptr};
  QPushButton * rotate_left_button_{nullptr};
  QPushButton * rotate_right_button_{nullptr};
  QPushButton * auto_start_button_{nullptr};
  QPushButton * auto_stop_button_{nullptr};
  QTimer * status_timer_{nullptr};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr auto_status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::mutex status_mutex_;
  std::mutex motion_mutex_;
  std::mutex auto_status_mutex_;
  std::string latest_status_;
  std::string latest_auto_status_;
  std::deque<MotionSample> motion_samples_;
  std::chrono::steady_clock::time_point last_odom_time_;
  bool has_odom_{false};
  bool motion_ready_{false};
  bool is_moving_{false};
  int motion_state_candidate_ticks_{0};
  double motion_speed_{0.0};
  double motion_yaw_rate_{0.0};
  std::string motion_style_class_;
  std::string auto_style_class_;
  bool approve_is_rotation_{false};
  bool approve_is_backup_{false};
};

}  // namespace scan_planner
