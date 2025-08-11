import numpy as np
import mujoco

from robot.robot import Robot
from environments.viewer import Viewer

class Environment:
    def __init__(self, mjcf_path, urdf_path = None):
        #mujoco robot
        self.robot = Robot(mjcf_path)
        self.viewer = Viewer()
        self.goal = None

    def get_state(self):
        return np.copy(self.robot.data.qpos)
    
    def set_goal(self, goal_pose):
        self.goal = goal_pose

    def get_goal(self):
        return self.goal
    
    def apply_action(self, q_target):
        for j in range(self.robot.n_dofs):
            self.robot.data.qpos[j] = q_target[j]#passes the joint position of the planner to the model
        mujoco.mj_forward(self.robot.model, self.robot.data)
                
    def run_viewer(self, trajectory):
        self.viewer.run_trajectory(self, trajectory)