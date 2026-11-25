# cmd_vel_sport_bridge

把 SCAN-Planner 的 `/cmd_vel` 转换成 Unitree Sport `Move` 请求：

- `linear.x -> Move.x`
- `linear.y -> Move.y`
- `angular.z -> Move.z`
- `geometry_msgs/msg/Twist` 的 `linear.z` 不使用

每条 `/cmd_vel` 立即发布一条 `/api/sport/request`，API ID 为 `1008`
（`ROBOT_SPORT_API_ID_MOVE`）。收到速度后，桥会在 `command_hold_time`
（默认 0.25 秒）内以 100 Hz 持续重发最近一条非零速度。Go2 的 Sport
`Move` 接口需要连续高频命令，单一低频命令只能产生约 1 cm 的位移；
持续重发可以兼容 10 Hz 的后退控制，也避免上游短时丢帧导致机器人停住。

Go2 的 ROS/DDS 只绑定 `eth0`，原生 Sport 服务也在这个网段上。因此直接发布
`/api/sport/request` 就是正确的入口，不需要旧的 Wi-Fi DDS relay。旧的
`native_wifi_relay_cpp.new` 只用于把狗端传感器暴露到 `wlan0`，会占用 Wi-Fi
带宽，SCAN-Planner 运行时不要启动它。

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/autonomy_stack_go2/install/local_setup.bash
source /home/unitree/SCAN-Planner/install/local_setup.bash

ros2 launch cmd_vel_sport_bridge cmd_vel_sport_bridge.launch.py
```

启动后可在 Go2 上用只绑定 `eth0` 的 DDS 配置检查：

```bash
ros2 topic info -v /api/sport/request
```

应看到 `Subscription count: 1`，即原生 Sport 服务的订阅者。

构建前必须先 source `autonomy_stack_go2`，因为 `unitree_api` 消息定义来自该工作空间。

超过 `command_timeout`（默认 0.5 秒）没有新的 `/cmd_vel` 时，桥会发送一次
零速度。停止节点本身不会发送 `StopMove`；应急停车应使用官方遥控器或
API `1003`，API `1001` 是 `Damp`，不是停车。
