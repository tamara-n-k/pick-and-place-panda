import numpy as np
import mujoco
import pinocchio as pin

from robot.mj_robot import MJRobot
from robot.pin_robot import PinRobot
from environments.viewer import Viewer


class Environment:
    """
    holds robot's states and provides utility functions for the task
    """
    def __init__(self, mjcf_path, urdf_path = None, mesh_dir = None):
        #mujoco robot
        self.robot = MJRobot(mjcf_path)
        self.pin = PinRobot(urdf_path, mesh_dir)
        self.mj_viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data) # passive viewer for visualization
        self.set_state(np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))

    def step(self):
        mujoco.mj_step(self.robot.model, self.robot.data)

    def set_state(self, pose):
        if len(pose) == 9:
            self.robot.data.qpos[:] = pose
        elif len(pose) == 7:
            self.robot.data.qpos[:7] = pose
        else:
            raise ValueError(f"Expected 7 or 9 joints, got {len(pose)}")
        self.step()

    def get_state(self):
        return np.copy(self.robot.data.qpos)

    def get_body_pos(self, body_name):
        return self.robot.data.body(body_name).xpos.copy()