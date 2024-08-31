# Implementation of Reinforcement Learning with Robots

"This project aims to implement the correct behavior for a robot in a simulation using Gazebo and ROS Noetic, leveraging machine learning frameworks such as TensorFlow and PyTorch for advanced perception tasks."

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Running the Project](#running-the-project)

## Prerequisites

Before you start, ensure you have the following software installed:

- **Ubuntu**: This project is intended to run on Ubuntu (20.04 mandatory).
- **Gazebo**: A robotics simulation software.
- **ROS Noetic**: A Robot Operating System (ROS) distribution that runs on Ubuntu 20.04.
- **TensorFlow 2.16**: A deep learning framework.
- **PyTorch**: Another popular deep learning framework.
- **CUDA 11.8**: A parallel computing platform and programming model by NVIDIA.
- **cuDNN 8.6.0**: A GPU-accelerated library for deep neural networks.
- **Numpy**: A library for numerical computing in Python.
- **Pandas**: A library for data manipulation and analysis in Python.

## Installation

### 1. Gazebo

Install Gazebo by running the following commands:

```bash
sudo apt-get update
sudo apt-get install gazebo11
```

### 2. ROS Noetic

Follow the official ROS Noetic installation guide for Ubuntu [here](https://wiki.ros.org/noetic/Installation/Ubuntu).


### 3. Additional ROS dependencies

Install additional ROS dependencies using the following command:

```bash
sudo apt install ros-noetic-image-transport ros-noetic-cv-bridge ros-noetic-vision-opencv python3-opencv libopencv-dev ros-noetic-image-proc
```

### 4. Tensorflow 2.16

Install TensorFlow 2.16 using pip:

```bash
pip install tensorflow==2.16.0
```

### 5. Pytorch

Install PyTorch. You can find the latest installation instructions on the [official PyTorch website](https://pytorch.org/get-started/previous-versions/). For CUDA 11.8 support, run the following:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. CUDA 11.8 and cuDNN 8.6.0

Ensure you have NVIDIA drivers installed. Then follow the steps on the official website to install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [cuDNN 8.6.0](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-860/install-guide/index.html).

Add CUDA to your path by updating your .bashrc file:
```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 7. Numpy and Pandas

Install Numpy and Pandas using pip:

```bash
pip install numpy pandas
```

## Setting Up the Environment

1. Setting up environment

First we need to create a dir where our code will be placed we can do the following

```bash
mkdir ~/catkin_ws
cd ~/catkin_ws
mkdir src
cd src
```

2. Download this repository

From the dir ~/catkin_ws/src we download the repository

```bash
git clone https://github.com/AlejandroNieto-DAM/TFG_UGR
```

We should get all the files inside the repository into the src folder.

3. Compile the project

```bash
cd ~/catkin_ws && catkin_make
```

## Running the Project

For executing the project and to simplify all we will open two terminals and we will use one for running the environment and another for running the algorithm.

### 1. First terminal

Here we need to export the turtlebot model that we will use. So we export an env variable like this:

```bash
export TURTLEBOT3_MODEL=burger
```

Once done that we can run the simulation with the next command:

```bash
cd ~/catkin_ws
roslaunch turtlebot3_gazebo turtlebot3_environmnet.launch stage:=x
```

We should replace the 'x' for the stage that we want we have from 1 to 4 stages.


### 2. Second terminal

Here we will launch the algorithm file so we need to export the next env variables:


```bash
export RL_ALGORITHM=DQN
export USING_CAMERA=0
export COINS=3
```

The variable 'RL_ALGORITHM' refers to the algorithm you want to run we have the implementations; DQN, PPO and SAC.
The variable 'USING_CAMERA' refers to if we are going to use the CNN nets or not, if the value is 0 we will use the ANN nets and if the value is 1 we will use the CNN nets.
The variable 'COINS' refers to the number if coins to place in the map.


Once we had finished the export of the variables we can run the algorithm with:

```bash
cd ~/catkin_ws
roslaunch turtlebot3_dqn turtlebot3_stage.launch stage:=x
```

We should replace the 'x' for the stage that we want we have from 1 to 4 stages (should match with the stage selected on the first terminal).

