# Beer Bot

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

## Build

```bash
colcon build --packages-select beer_bot
source install/setup.bash
colcon test --packages-select beer_bot
```

## Run

```bash
ros2 launch beer_bot bringup_sim.launch.py   # simulation
ros2 launch beer_bot bringup_pi.launch.py    # on-device
```
