import numpy as np
import mujoco
import pinocchio as pin

from robot.robot import Robot
from robot.pin_robot import PinRobot
from environments.viewer import Viewer


class Environment:
    """
    holds robot's states: current position, goal position
    """
    def __init__(self, mjcf_path, urdf_path = None, mesh_dir = None):
        #mujoco robot
        self.robot = Robot(mjcf_path)
        self.goal = None
        self.pin = PinRobot(urdf_path, mesh_dir)
        self.viewer = Viewer
        OPEN_GRIPPER = np.array([0.04, 0.04])
        CLOSED_GRIPPER = np.array([0, 0])
        self.gripper = OPEN_GRIPPER
        self.set_state(np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))

    def step(self):
        mujoco.mj_step(self.robot.model, self.robot.data)

    def set_state(self, pose):
        self.robot.data.qpos = np.concatenate((pose, self.gripper))
        self.step()

    def get_state(self):
        return np.copy(self.robot.data.qpos)
    
    def set_goal_joint(self, goal_pose):
        self.goal = np.concatenate((goal_pose, self.gripper))
    
    def set_goal_cartesian(self, pos: np.ndarray, rot: np.ndarray = None):
        R = rot if rot is not None else pin.utils.rpyToMatrix(0.0, np.pi, 0.0)#pitch = pi --> face downwards
        self.goal_cartesian = pin.SE3(R, np.asarray(pos))

    def get_goal(self):
        return self.goal

    def get_body_pos(self, body_name):
        return self.robot.data.body(body_name).xpos.copy()