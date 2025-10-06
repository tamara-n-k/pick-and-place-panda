# Franka Emika Panda Simulation
## Overview

Goal: simulation of a panda robot for manipulation of objects

## Setup
Ubuntu-Version 24.04
1. install mujoco via pip
2. install ompl with python wheels
pip install wheel
https://ompl.kavrakilab.org/installation.html
pip install ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl 
OR use environment.yml

## Files
- udrf file
- mujoco xml
- assets

## Software Requirements

## Code Arquitecture

<p float="left">
  <img src="panda_mujoco_sim.png" width="400">
</p>


Environment:
- owns Robot & PinRobot
- holds current state & goal
- exposes apply_action() to update robot's state

Planner:
- gets current state + goal from Environment
- computes action sequence (joint configs trajectory)
- returns that sequence

Robot:
- holds the robot's model information in MJCF format

PinRobot:
- holds the robot's model information in URDF format

IKSolver:
- uses the robot's urdf model and target position to get an IK solution

Main:
- create Environment
- create Planner
- call planner with env info
- send planned actions back to env to apply
- env’s Viewer visualizes it

Cartesian goal → IKPlanner → q_goal → RRTPlanner → joint trajectory → Viewer


## Some Theory

Joint ID vs. Frame ID:

    A joint describes a degree of freedom in the kinematic chain (rotation, translation, etc.).

    A frame is a coordinate system attached to a joint, link, or custom location in your robot model.