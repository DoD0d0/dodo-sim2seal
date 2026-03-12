# Dodo Sim2Sim: RL Policy Deployment in Isaac Sim

ROS2-based deployment framework for running IsaacLab-trained reinforcement learning policies on the Dodo biped robot in NVIDIA Isaac Sim.

## Overview

The pipeline splits across two environments:

- **Host**: NVIDIA Isaac Sim 5.1.0 — runs the physics simulation, publishes `/imu` and `/joint_states`, subscribes to `/joint_command`
- **Docker container**: ROS2 Jazzy — runs the RL policy controller node, reads sensor data, outputs joint targets

Communication is over `--network host` with `ROS_DOMAIN_ID=0`.

## System Requirements

- Ubuntu 24.04 LTS
- NVIDIA Isaac Sim 5.1.0 (installed on host)
- Docker with NVIDIA Container Toolkit (optional, only needed if GPU inference is required)
- Python 3.12

## Robot Specifications

**Dodo Biped Robot**

| Property | Value |
|----------|-------|
| Degrees of Freedom | 8 (4 per leg) |
| Control Frequency | 50 Hz |
| Physics Timestep | 5 ms (0.005s) |
| Policy Decimation | 4 |
| Actuator Type | PD Controller |
| Stiffness | 40 N⋅m/rad |
| Damping | 2 N⋅m⋅s/rad |

**Joint order**: `left_joint_1–4`, then `right_joint_1–4`

## Repository Structure

```
dodo-sim2seal/
├── build_ros.sh                      # Docker build helper script
├── dockerfiles/                      # Container build definitions
└── jazzy_ws/
    ├── assets/dodo/
    │   ├── urdf/dodo.urdf
    │   └── usd/
    │       ├── dodo.usd              # Base USD model
    │       └── dodo_simple_ROS.usd   # USD with OmniGraph (use this)
    ├── model/
    │   ├── dodo_env.yaml             # Environment configuration
    │   └── isaaclab/
    │       ├── stand/                # Stand policy (.pt)
    │       ├── walk/                 # Walk policy (.pt)
    │       └── jump/                 # Jump policy (.pt)
    └── src/
        ├── dodo_controller/          # RL policy controller node
        ├── isaacsim/                 # Isaac Sim launcher node
        └── isaac_ros2_messages/      # Custom service definitions
```

## Quick Start

### Step 1: Build Docker image

```bash
cd ~/TUM/2025w/dodo/dodo-sim2seal

docker build -t dodo-sim2seal:jazzy \
  -f dockerfiles/ubuntu_24_jazzy_python_312_minimal.dockerfile .
```

### Step 2: Start Docker container

The controller node runs on CPU — no GPU flag needed for this container.

```bash
docker run --rm -it \
  --name dodo-sim2seal-jazzy \
  --network host \
  -e ROS_DOMAIN_ID=0 \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/repo/jazzy_ws/fastdds.xml \
  -v $(pwd):/repo \
  -w /repo \
  dodo-sim2seal:jazzy \
  bash
```

To use the docker container which is already exist:

```bash
# To start new
docker start -ai dodo-sim2seal-jazzy
# To open another terminal
docker exec -it dodo-sim2seal-jazzy bash
```

### Step 3: Build ROS2 workspace inside container

```bash
# Source base ROS packages built into the image
source /workspace/jazzy_ws/install/setup.bash

# Build the dodo packages from the mounted repo
cd /repo/jazzy_ws
colcon build
source install/setup.bash

# Verify packages are available
ros2 pkg list | grep dodo
# Or
ros2 pkg list
```

Expected output:
```
dodo_controller
isaac_ros2_messages
```

### Step 4: Launch policy controller first (inside container)

The controller must be running **before** Isaac Sim starts playing. This ensures `/joint_command` is being published when the physics simulation begins.

```bash
ros2 launch dodo_controller dodo_controller.launch.py policy_type:=stand
```

### Step 5: Launch Isaac Sim (on host)

Isaac Sim runs on the **host**, not inside Docker. Default install path for Isaac Sim 5.1.0 is `~/isaac-sim`.

#### Option A: Shell command (always works)

Open a new terminal on the host:

```bash
~/isaac-sim/isaac-sim.sh \
  --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```

Once the GUI opens:
1. `File → Open` → select `jazzy_ws/assets/dodo/usd/dodo_simple_ROS.usd`
2. Click **Play** (spacebar or the triangle button in the toolbar)

#### Option B: ROS2 launch (auto-loads USD and presses Play)

Requires ROS2 Jazzy on the host. Run in a new host terminal:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/TUM/2025w/dodo/dodo-sim2seal/jazzy_ws
colcon build --packages-select isaacsim isaac_ros2_messages
source install/setup.bash

ros2 launch isaacsim run_isaacsim.launch.py \
  gui:=$(pwd)/assets/dodo/usd/dodo_simple_ROS.usd \
  stage_script:=$(pwd)/src/isaacsim/scripts/open_dodo_stage.py \
  play_sim_on_start:=true
```

If Isaac Sim is installed in a non-default location, add `install_path:=/path/to/isaacsim`.

#### Verify Isaac Sim is publishing (inside container)

```bash
ros2 topic list
```

Expected:
```
/clock
/imu
/joint_states
```

If `/clock` is missing, the ROS2 bridge is not active. Re-launch with the `--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge` flag (Option A) or enable it manually in Isaac Sim via `Window → Extensions → isaacsim.ros2.bridge`.

### Step 6: Verify control output

```bash
# Joint commands should appear at ~50 Hz
ros2 topic hz /joint_command

# Check joint names and values
ros2 topic echo /joint_command
```

### Step 7: Send velocity commands

```bash
# Zero command (stand still)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}" --once

# Forward walk
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.0}}" --once

# Turn
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.5}}" --once
```

## Policy Specifications

### Stand / Walk

**Observation (36 dimensions):**

| Component | Dims |
|-----------|------|
| Linear velocity (body frame) | 3 |
| Angular velocity (body frame) | 3 |
| Gravity direction (body frame) | 3 |
| Velocity command | 3 |
| Joint positions (relative to default) | 8 |
| Joint velocities | 8 |
| Previous action | 8 |

Action scale: `0.5`

### Jump

**Observation (191 dimensions):**

| Component | Dims |
|-----------|------|
| Gravity direction (body frame) | 3 |
| Target position (body frame) | 3 |
| Time to target | 1 |
| Joint positions (relative) | 8 |
| Joint velocities | 8 |
| Previous action | 8 |
| Height scan (16×10 grid, 0.1m res) | 160 |

Action scale: `0.45`

Additional requirements: height scanner sensor, box obstacle in scene, `/jump_target`, `/jump_time`, `/height_scan` publishers.

## ROS2 Interface

### Subscribed Topics

| Topic | Type | Used by |
|-------|------|---------|
| `/imu` | `sensor_msgs/Imu` | stand, walk, jump |
| `/joint_states` | `sensor_msgs/JointState` | stand, walk, jump |
| `/cmd_vel` | `geometry_msgs/Twist` | stand, walk |
| `/jump_target` | `geometry_msgs/Point` | jump |
| `/jump_time` | `std_msgs/Float32` | jump |
| `/height_scan` | `std_msgs/Float32MultiArray` | jump |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_command` | `sensor_msgs/JointState` | Commanded joint positions |

> `/odom` (`nav_msgs/Odometry`) is subscribed if available and used for body-frame linear velocity. If absent, falls back to scaled `cmd_vel`.

## Launch Parameters

### `dodo_controller.launch.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy_type` | `stand` | `stand`, `walk`, or `jump` |
| `policy_path` | `""` | Explicit `.pt` path (auto-discovers if empty) |
| `action_scale` | `-1.0` | Action scale (-1 = auto: 0.5 for stand/walk, 0.45 for jump) |
| `decimation` | `4` | Steps between policy inference calls |
| `training_joint_order` | `right_left` | Joint order used during training: `right_left`, `left_right`, or `isaac` |
| `startup_ramp_sec` | `1.0` | Ramp action from 0→1 over this many seconds at startup |
| `hold_default_pose_sec` | `0.5` | Hold default pose before policy control begins |
| `max_policy_action_abs` | `2.0` | Clip raw policy output magnitude |
| `max_action_delta` | `0.15` | Maximum per-step action change |
| `cmd_vel_to_lin_vel_gain` | `0.35` | Scale factor for cmd_vel fallback when `/odom` is missing |

### `dodo_sim2sim.launch.py`

Launches Isaac Sim node + controller together (requires Isaac Sim installed on host).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_path` | `""` | Path to USD scene file |
| `policy_type` | `stand` | Policy behavior |
| `policy_path` | `""` | Explicit policy path |
| `version` | `5.1.0` | Isaac Sim version |
| `base_height` | `0.575` | Initial robot base height (m) |
| `joint_stiffness` | `40.0` | PD stiffness applied at stage load |
| `joint_damping` | `2.0` | PD damping applied at stage load |

## Troubleshooting

### Docker GPU error (`nvml error: driver not loaded`)

The controller node does not require GPU. Remove `--gpus all` from the `docker run` command. Isaac Sim runs on the host and uses the GPU directly.

If GPU inference is needed inside the container, ensure `nvidia-container-toolkit` is installed and configured on the host:
```bash
nvidia-smi          # verify driver is loaded on host
sudo systemctl restart docker
```

### Policy file not found

```bash
find /repo/jazzy_ws/model/isaaclab -name "policy.pt"
```

Pass the path explicitly via `policy_path:=...`.

### `/clock` not published

Isaac Sim ROS2 bridge is not running. In Isaac Sim:
- Window → Extensions → search `isaacsim.ros2.bridge` → enable
- Press Play

### colcon build fails (`--uninstall not recognized`)

```bash
rm -rf jazzy_ws/build/dodo_controller jazzy_ws/install/dodo_controller
cd jazzy_ws && colcon build
```

### Joint order mismatch

Check controller startup log for DOF names. If the Isaac Sim DOF order differs from the training order, set `training_joint_order` launch argument accordingly (`right_left`, `left_right`, or `isaac`).

## References

- IsaacLab: https://isaac-sim.github.io/IsaacLab/
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac-sim
- ROS2 Jazzy: https://docs.ros.org/en/jazzy/

## License

Apache License 2.0

## Authors

TUM Dodo RL Team, 2025W