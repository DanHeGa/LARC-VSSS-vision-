# LARC-VSSS-vision-

ROS2 implementation of the VSSS vision system for LARC International

## Overview

This repository provides a vision system for the Very Small Size Soccer (VSSS) league, implemented using ROS2 (Robot Operating System 2). It is designed for use in the LARC International competition, focusing on real-time detection and tracking of robots and the ball.

## Features

- 100% Python implementation
- Real-time image processing for robot soccer
- Integration with ROS2 nodes and messaging
- Modular and extensible for custom vision algorithms

## Getting Started

### Prerequisites

- ROS2 (HUMBLE)
- Python 3.x
- [Other dependencies, e.g., OpenCV, NumPy, etc.]

### Installation

Clone this repository:

```bash
git clone https://github.com/DanHeGa/LARC-VSSS-vision-.git
```
### Setup
From your ROS workspace run:

```bash
colcon build --symlink-install

source install\setup.bash
```
### Usage
Without using a launch file, in order to run the model with your desired camera input, run the next commands:

Node to get raw camera input
```bash
ros2 run vision camera_input
#Using parameters to set the input video id: 
ros2 run vision camera_input --ros-args -p Video_ID:=0
```
Node to get the field warped image
```bash
ros2 run vision image_warp
```
Node to use the model in warped images
```bash
ros2 run vision model_use
```

Update the package and node name as appropriate for your implementation.

## Folder Structure

```
/
├── src/
  ├── resource
  ├── test
  ├── utils
  ├── vision
├── launch/             # ROS2 launch files
├── .gitignore
├── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, feature requests, or improvements.


## Contact

For questions or support, contact [A00838702@tec.mx] or open an issue in this repository.
