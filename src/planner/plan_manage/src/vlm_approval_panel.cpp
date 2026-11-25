#include "plan_manage/vlm_approval_panel.hpp"

#include <cmath>
#include <sstream>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <json/json.h>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace scan_planner
{

namespace
{
double wrapAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
}  // namespace

VlmApprovalPanel::VlmApprovalPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout(this);

  auto * title = new QLabel("VLM 导航审批", this);
  QFont title_font = title->font();
  title_font.setBold(true);
  title->setFont(title_font);
  layout->addWidget(title);

  motion_label_ = new QLabel("机械狗状态：等待里程计", this);
  motion_label_->setAlignment(Qt::AlignCenter);
  motion_label_->setMinimumHeight(38);
  motion_label_->setWordWrap(true);
  motion_label_->setStyleSheet(
    "QLabel { background-color: #555555; color: white; "
    "font-weight: bold; padding: 6px; }");
  layout->addWidget(motion_label_);

  auto_label_ = new QLabel("自动执行：未启动", this);
  auto_label_->setAlignment(Qt::AlignCenter);
  auto_label_->setMinimumHeight(34);
  auto_label_->setWordWrap(true);
  auto_label_->setStyleSheet(
    "QLabel { background-color: #444444; color: white; "
    "font-weight: bold; padding: 6px; }");
  layout->addWidget(auto_label_);

  status_label_ = new QLabel(
    "等待 VLM 候选点。候选生成后不会自动运动。", this);
  status_label_->setWordWrap(true);
  status_label_->setMinimumHeight(105);
  layout->addWidget(status_label_);

  approve_button_ = new QPushButton("批准并发送到 SCAN", this);
  approve_button_->setStyleSheet(
    "QPushButton { background-color: #218c3a; color: white; "
    "font-weight: bold; padding: 8px; }");
  approve_button_->setEnabled(false);
  layout->addWidget(approve_button_);

  reselect_button_ = new QPushButton("让 VLM 重选", this);
  reselect_button_->setStyleSheet(
    "QPushButton { background-color: #b35c00; color: white; "
    "font-weight: bold; padding: 8px; }");
  layout->addWidget(reselect_button_);

  backup_button_ = new QPushButton("后退 0.3 m", this);
  backup_button_->setStyleSheet(
    "QPushButton { background-color: #8a5a12; color: white; "
    "font-weight: bold; padding: 8px; }");
  backup_button_->setEnabled(false);
  layout->addWidget(backup_button_);

  auto * rotate_layout = new QHBoxLayout();
  rotate_left_button_ = new QPushButton("左转 30°", this);
  rotate_left_button_->setStyleSheet(
    "QPushButton { background-color: #4f6d8a; color: white; "
    "font-weight: bold; padding: 8px; }");
  rotate_right_button_ = new QPushButton("右转 30°", this);
  rotate_right_button_->setStyleSheet(
    "QPushButton { background-color: #6d4f8a; color: white; "
    "font-weight: bold; padding: 8px; }");
  rotate_layout->addWidget(rotate_left_button_);
  rotate_layout->addWidget(rotate_right_button_);
  layout->addLayout(rotate_layout);

  auto_start_button_ = new QPushButton("开始自动执行", this);
  auto_start_button_->setStyleSheet(
    "QPushButton { background-color: #1769aa; color: white; "
    "font-weight: bold; padding: 8px; }");
  layout->addWidget(auto_start_button_);

  auto_stop_button_ = new QPushButton("结束自动执行", this);
  auto_stop_button_->setStyleSheet(
    "QPushButton { background-color: #9b2c2c; color: white; "
    "font-weight: bold; padding: 8px; }");
  auto_stop_button_->setEnabled(false);
  layout->addWidget(auto_stop_button_);
  layout->addStretch();

  connect(approve_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::approveGoal);
  connect(reselect_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::requestReselect);
  connect(backup_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::backupDistance);
  connect(rotate_left_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::rotateLeft30);
  connect(rotate_right_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::rotateRight30);
  connect(auto_start_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::startAutoExecution);
  connect(auto_stop_button_, &QPushButton::clicked,
    this, &VlmApprovalPanel::stopAutoExecution);

  status_timer_ = new QTimer(this);
  status_timer_->setInterval(150);
  connect(status_timer_, &QTimer::timeout,
    this, &VlmApprovalPanel::refreshStatus);
  status_timer_->start();
}

void VlmApprovalPanel::onInitialize()
{
  auto ros_node = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!ros_node) {
    status_label_->setText("错误：RViz ROS 节点不可用。");
    approve_button_->setEnabled(false);
    reselect_button_->setEnabled(false);
    backup_button_->setEnabled(false);
    rotate_left_button_->setEnabled(false);
    rotate_right_button_->setEnabled(false);
    auto_start_button_->setEnabled(false);
    auto_stop_button_->setEnabled(false);
    return;
  }

  auto node = ros_node->get_raw_node();
  command_pub_ = node->create_publisher<std_msgs::msg::String>(
    "/go2/vlm_approval", 10);
  rclcpp::QoS status_qos(1);
  status_qos.transient_local().reliable();
  status_sub_ = node->create_subscription<std_msgs::msg::String>(
    "/vlm_pixel_shadow/status", status_qos,
    [this](std_msgs::msg::String::ConstSharedPtr message) {
      std::lock_guard<std::mutex> guard(status_mutex_);
      latest_status_ = message->data;
    });
  auto_status_sub_ = node->create_subscription<std_msgs::msg::String>(
    "/vlm_pixel_shadow/auto_status", status_qos,
    [this](std_msgs::msg::String::ConstSharedPtr message) {
      std::lock_guard<std::mutex> guard(auto_status_mutex_);
      latest_auto_status_ = message->data;
    });
  rclcpp::QoS odom_qos(10);
  odom_qos.best_effort();
  odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
    "/LIO/odom_vehicle", odom_qos,
    [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
      odomCallback(message);
    });
  backup_button_->setEnabled(true);
  rotate_left_button_->setEnabled(true);
  rotate_right_button_->setEnabled(true);
  auto_start_button_->setEnabled(true);
}

void VlmApprovalPanel::publishCommand(const std::string & command)
{
  if (!command_pub_) {
    status_label_->setText("错误：审批命令发布器未初始化。");
    return;
  }
  std_msgs::msg::String message;
  message.data = command;
  command_pub_->publish(message);
}

void VlmApprovalPanel::odomCallback(
  const nav_msgs::msg::Odometry::ConstSharedPtr & message)
{
  const auto now = std::chrono::steady_clock::now();
  const auto & pose = message->pose.pose;
  const MotionSample sample{
    now,
    pose.position.x,
    pose.position.y,
    yawFromQuaternion(pose.orientation),
  };

  std::lock_guard<std::mutex> guard(motion_mutex_);
  if (
    has_odom_ &&
    now - last_odom_time_ > std::chrono::milliseconds(500))
  {
    motion_samples_.clear();
  }
  has_odom_ = true;
  last_odom_time_ = now;
  motion_samples_.push_back(sample);
  while (
    !motion_samples_.empty() &&
    now - motion_samples_.front().time > std::chrono::milliseconds(1000))
  {
    motion_samples_.pop_front();
  }
  if (motion_samples_.size() < 2) {
    motion_ready_ = false;
    return;
  }

  const auto & oldest = motion_samples_.front();
  const double span = std::chrono::duration<double>(
    now - oldest.time).count();
  if (span < 0.7) {
    motion_ready_ = false;
    return;
  }
  const double distance = std::hypot(
    sample.x - oldest.x,
    sample.y - oldest.y);
  const double yaw_change = std::abs(
    wrapAngle(sample.yaw - oldest.yaw));
  motion_speed_ = distance / span;
  motion_yaw_rate_ = yaw_change / span;
  is_moving_ =
    motion_speed_ > 0.03 ||
    motion_yaw_rate_ > 0.05;
  motion_ready_ = true;
}

void VlmApprovalPanel::updateMotionLabel()
{
  bool has_odom;
  bool motion_ready;
  bool is_moving;
  double speed;
  double yaw_rate;
  std::chrono::steady_clock::time_point last_odom_time;
  {
    std::lock_guard<std::mutex> guard(motion_mutex_);
    has_odom = has_odom_;
    motion_ready = motion_ready_;
    is_moving = is_moving_;
    speed = motion_speed_;
    yaw_rate = motion_yaw_rate_;
    last_odom_time = last_odom_time_;
  }

  const auto now = std::chrono::steady_clock::now();
  std::string style_class;
  QString text;
  if (
    !has_odom ||
    now - last_odom_time > std::chrono::milliseconds(500))
  {
    style_class = "unknown";
    text = "机械狗状态：状态未知（里程计超时）";
  } else if (!motion_ready) {
    style_class = "settling";
    text = "机械狗状态：判定中…";
  } else if (is_moving) {
    style_class = "moving";
    text = QString("机械狗状态：运动中  v=%1 m/s  ω=%2 rad/s")
      .arg(speed, 0, 'f', 2)
      .arg(yaw_rate, 0, 'f', 2);
  } else {
    style_class = "stationary";
    text = QString("机械狗状态：停稳  v=%1 m/s  ω=%2 rad/s")
      .arg(speed, 0, 'f', 2)
      .arg(yaw_rate, 0, 'f', 2);
  }
  motion_label_->setText(text);
  if (style_class == motion_style_class_) {
    return;
  }
  motion_style_class_ = style_class;
  if (style_class == "stationary") {
    motion_label_->setStyleSheet(
      "QLabel { background-color: #218c3a; color: white; "
      "font-weight: bold; padding: 6px; }");
  } else if (style_class == "moving") {
    motion_label_->setStyleSheet(
      "QLabel { background-color: #b35c00; color: white; "
      "font-weight: bold; padding: 6px; }");
  } else if (style_class == "settling") {
    motion_label_->setStyleSheet(
      "QLabel { background-color: #8a6d1d; color: white; "
      "font-weight: bold; padding: 6px; }");
  } else {
    motion_label_->setStyleSheet(
      "QLabel { background-color: #555555; color: white; "
      "font-weight: bold; padding: 6px; }");
  }
}

void VlmApprovalPanel::approveGoal()
{
  if (!approve_button_->isEnabled()) {
    status_label_->setText("当前没有待批准的 VLM 候选点。");
    return;
  }
  publishCommand("approve");
  status_label_->setText(
    approve_is_backup_
    ? "已提交批准，等待 VLM 节点执行后退。"
    : (approve_is_rotation_
      ? "已提交批准，等待 VLM 节点执行旋转。"
      : "已提交批准，等待 VLM 节点发送 SCAN 目标。"));
}

void VlmApprovalPanel::requestReselect()
{
  publishCommand("reselect");
  status_label_->setText("已请求重选，正在等待新的 VLM 候选点。");
}

void VlmApprovalPanel::backupDistance()
{
  publishCommand("backup");
  status_label_->setText(
    "已请求后退，正在按后方净距限制最大 0.3 m。");
}

void VlmApprovalPanel::rotateLeft30()
{
  publishCommand("rotate_left_30");
  status_label_->setText("已请求左转 30°。");
}

void VlmApprovalPanel::rotateRight30()
{
  publishCommand("rotate_right_30");
  status_label_->setText("已请求右转 30°。");
}

void VlmApprovalPanel::startAutoExecution()
{
  publishCommand("auto_start");
  auto_label_->setText("自动执行：已提交启动请求");
}

void VlmApprovalPanel::stopAutoExecution()
{
  publishCommand("auto_stop");
  auto_label_->setText(
    "自动执行：结束请求已发送，当前动作完成后停止。");
}

void VlmApprovalPanel::refreshAutoStatus()
{
  std::string status;
  {
    std::lock_guard<std::mutex> guard(auto_status_mutex_);
    status = latest_auto_status_;
  }
  if (status.empty()) {
    auto_label_->setText("自动执行：未启动");
    auto_start_button_->setEnabled(true);
    auto_stop_button_->setEnabled(false);
    return;
  }

  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::istringstream stream(status);
  if (!Json::parseFromStream(builder, stream, &root, &errors)) {
    auto_label_->setText(QString::fromStdString(status));
    return;
  }

  const bool active = root.get("auto_active", false).asBool();
  const bool stop_requested =
    root.get("auto_stop_requested", false).asBool();
  const std::string message =
    root.get("auto_message", "").asString();
  std::string style_class;
  if (!active) {
    style_class = "off";
    auto_label_->setText(
      message.empty()
      ? QString("自动执行：未启动")
      : QString("自动执行：已停止 · ")
        + QString::fromStdString(message));
  } else if (stop_requested) {
    style_class = "stopping";
    auto_label_->setText(
      QString("自动执行：结束中 · ")
      + QString::fromStdString(message));
  } else {
    style_class = "active";
    auto_label_->setText(
      QString("自动执行：运行中 · ")
      + QString::fromStdString(message));
  }
  auto_start_button_->setEnabled(!active);
  auto_stop_button_->setEnabled(active && !stop_requested);

  if (style_class == auto_style_class_) {
    return;
  }
  auto_style_class_ = style_class;
  if (style_class == "active") {
    auto_label_->setStyleSheet(
      "QLabel { background-color: #1769aa; color: white; "
      "font-weight: bold; padding: 6px; }");
  } else if (style_class == "stopping") {
    auto_label_->setStyleSheet(
      "QLabel { background-color: #8a6d1d; color: white; "
      "font-weight: bold; padding: 6px; }");
  } else {
    auto_label_->setStyleSheet(
      "QLabel { background-color: #444444; color: white; "
      "font-weight: bold; padding: 6px; }");
  }
}

void VlmApprovalPanel::refreshStatus()
{
  updateMotionLabel();
  refreshAutoStatus();

  std::string status;
  {
    std::lock_guard<std::mutex> guard(status_mutex_);
    status = latest_status_;
  }
  if (status.empty()) {
    return;
  }

  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::istringstream stream(status);
  if (!Json::parseFromStream(builder, stream, &root, &errors)) {
    approve_button_->setEnabled(false);
    status_label_->setText(QString::fromStdString(status));
    return;
  }

  const bool approval_required =
    root.get("approval_required", false).asBool();
  const bool rotation_request =
    approval_required &&
    root.get("action", "").asString() == "rotate";
  const bool backup_request =
    approval_required &&
    root.get("action", "").asString() == "backup";
  approve_is_rotation_ = rotation_request;
  approve_is_backup_ = backup_request;
  approve_button_->setEnabled(approval_required);
  approve_button_->setText(
    backup_request
    ? "批准并执行后退"
    : (rotation_request
      ? "批准并执行旋转"
      : "批准并发送到 SCAN"));
  std::ostringstream display;
  if (root.get("backup_started", false).asBool()) {
    display << "正在后退 "
            << root.get("backup_distance_m", 0.0).asFloat()
            << " m（后方净距 "
            << root.get("backup_rear_clearance_m", 0.0).asFloat()
            << " m）。\n";
  } else if (root.isMember("backup_complete")) {
    display << (root["backup_complete"].asBool()
      ? "后退完成，正在重新选点。\n"
      : "后退未完成。\n");
  }
  if (approval_required) {
    if (backup_request) {
      display << "已计算可后退距离: "
              << root.get("backup_distance_m", 0.0).asFloat()
              << " m\n后方净距: "
              << root.get("backup_rear_clearance_m", 0.0).asFloat()
              << " m\n点击批准执行后退。\n";
    } else if (rotation_request) {
      display << "VLM 判定当前视角无候选点。\n";
      const double angle = root.get("rotate_deg", 0.0).asFloat();
      display << "建议原地"
              << (angle >= 0.0 ? "左转 " : "右转 ")
              << std::abs(angle) << "°\n";
    } else {
      display << "等待审批，狗保持停止。\n";
    }
    if (root.isMember("pixel") && root["pixel"].isArray()) {
      display << "像素: [";
      for (Json::ArrayIndex index = 0; index < root["pixel"].size(); ++index) {
        if (index > 0) {
          display << ", ";
        }
        display << root["pixel"][index].asInt();
      }
      display << "]\n";
    }
    if (root.isMember("target_xyz") && root["target_xyz"].isArray()) {
      display << "世界点: [";
      for (Json::ArrayIndex index = 0;
        index < root["target_xyz"].size() && index < 2; ++index)
      {
        if (index > 0) {
          display << ", ";
        }
        display << root["target_xyz"][index].asFloat();
      }
      display << "]\n";
    }
    if (root.isMember("reason")) {
      display << "理由: " << root["reason"].asString();
    }
  } else if (root.get("goal_publishing", false).asBool()) {
    display << "已批准，目标已发送给 SCAN。\n";
    if (root.isMember("target_xyz") && root["target_xyz"].isArray()) {
      display << "目标: [";
      for (Json::ArrayIndex index = 0;
        index < root["target_xyz"].size() && index < 2; ++index)
      {
        if (index > 0) {
          display << ", ";
        }
        display << root["target_xyz"][index].asFloat();
      }
      display << "]";
    }
  } else if (root.get("reselecting", false).asBool()) {
    display << "已解除锁定，VLM 正在重新选点。";
  } else if (root.get("rotation_started", false).asBool()) {
    display << "正在原地旋转 "
            << std::abs(root.get("rotate_deg", 0.0).asFloat())
            << "°。";
  } else if (root.get("action", "").asString() == "rotate_done") {
    display << "旋转完成，正在重新调用 VLM 选点。";
  } else if (!root.get("ok", true).asBool()) {
    display << "VLM 查询失败: "
            << root.get("error", "unknown error").asString();
  } else if (!root.get("walkable", true).asBool()) {
    display << "VLM 判定当前不可走，等待下次重试。";
  } else {
    display << "VLM 正在生成候选点。";
  }
  status_label_->setText(QString::fromStdString(display.str()));
}

}  // namespace scan_planner

PLUGINLIB_EXPORT_CLASS(scan_planner::VlmApprovalPanel, rviz_common::Panel)
