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
        self.mj_viewer = None
        self.mj_viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data) # passive viewer for visualization
        self.robot_base_pos = self.robot.data.body('link0').xpos.copy()
        print(f"Robot base position: {self.robot_base_pos}")
        self.robot_base_mat = self.robot.data.body('link0').xmat.reshape(3, 3).copy()

    def step(self):
        ### ---gravity compensation for arm joints: gravity, coriolis, centrifugal forces --- ###
         # clear any existing forces from previous steps
        self.robot.data.qfrc_applied[:] = 0
         # apply compensationtorque equal to mujoco's internal gravity compensation for the arm joints
        self.robot.data.qfrc_applied[:7] = self.robot.data.qfrc_bias[:7]
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
        # reset velocities to zero for stability
        self.robot.data.qvel[:] = 0
        self.robot.data.qacc[:] = 0
        # set arm and gripper controls to match the new state
        self.robot.set_arm_ctrl(self.robot.data.qpos[:7])
        if self.robot.model.nu > 7:
            self.robot.set_gripper_ctrl(np.sum(self.robot.data.qpos[7:9]))

        mujoco.mj_forward(self.robot.model, self.robot.data)
        if hasattr(self, 'pin'):
            self.pin.update_state(self.robot.data.qpos[:7]) # update pinocchio geometry for collision checking

        if self.mj_viewer is not None and self.mj_viewer.is_running():
            self.mj_viewer.sync()

    # simple open loop servo function to execute planned trajectories with position control
    def servo_to(self, q_arm, gripper_width=None, steps=400, tolerance=0.01):
        q_arm = np.asarray(q_arm)
        self.robot.set_arm_ctrl(q_arm)
        if gripper_width is not None:
            self.robot.set_gripper_ctrl(gripper_width)

        for _ in range(steps):
            self.step()
            if self.mj_viewer is not None and self.mj_viewer.is_running():
                self.mj_viewer.sync()
            if np.linalg.norm(self.robot.data.qpos[:7] - q_arm) < tolerance:
                break # early exit if we are close enough to the target to save time

        return np.linalg.norm(self.robot.data.qpos[:7] - q_arm) < tolerance


    def get_arm_state(self):
        return np.copy(self.robot.data.qpos[:7])



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
