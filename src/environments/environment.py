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
        self.set_state(self.robot.home_pose)
        self.robot_base_pos = self.robot.data.body('link0').xpos.copy()
        print(f"Robot base position: {self.robot_base_pos}")
        self.robot_base_mat = self.robot.data.body('link0').xmat.reshape(3, 3).copy()

    def step(self):
        mujoco.mj_step(self.robot.model, self.robot.data)

    def set_state(self, pose):
        pose = np.asarray(pose)

        if len(pose) == self.robot.model.nq:
            self.robot.data.qpos[:] = pose
        elif len(pose) == 9:
            self.robot.data.qpos[:9] = pose
        elif len(pose) == 7:
            self.robot.data.qpos[:7] = pose
        else:
            raise ValueError(f"Expected 7, 9, or {self.robot.model.nq} qpos values, got {len(pose)}")
        self.robot.data.qvel[:] = 0
        self.robot.data.qacc[:] = 0

        mujoco.mj_forward(self.robot.model, self.robot.data)
        if hasattr(self, 'pin'):
            self.pin.update_state(self.robot.data.qpos[:7]) # update pinocchio geometry for collision checking

        if self.mj_viewer.is_running():
            self.mj_viewer.sync()


    def get_arm_state(self):
        return np.copy(self.robot.data.qpos[:7])


    def get_full_state(self):
        return np.copy(self.robot.data.qpos)


    def get_gripper_state(self):
        return np.copy(self.robot.data.qpos[7:9])


    def get_state(self):
        return self.get_arm_state()


    def get_body_pos(self, body_name):
        world_pos = self.robot.data.body(body_name).xpos
        # 1. Translate
        centered_pos = world_pos - self.robot_base_pos
        # 2. Rotate into the robot's frame using the transpose (inverse) of base matrix
        relative_pos = self.robot_base_mat.T @ centered_pos
        print(f"World position of {body_name}: {world_pos}, Relative position: {relative_pos}")
        return relative_pos
    

    def get_place_pose(self, tray_name, hole_offset_local, target_box_rotation):
        tray_world_pos = self.robot.data.body(tray_name).xpos
        tray_world_mat = self.robot.data.body(tray_name).xmat.reshape(3, 3)
        
        hole_world_pos = tray_world_pos + (tray_world_mat @ hole_offset_local)
        
        centered_pos = hole_world_pos - self.robot_base_pos
        hole_relative_pos = self.robot_base_mat.T @ centered_pos
        
        hole_relative_mat = self.robot_base_mat.T @ target_box_rotation
        
        place_SE3 = pin.SE3(hole_relative_mat, hole_relative_pos)
        
        print(f"Relative Base Pos packed into SE3: {hole_relative_pos}")
        return place_SE3