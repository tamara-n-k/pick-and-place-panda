import mujoco
import numpy as np

class MJRobot:
    """
    holds the robot model data for the mujoco simulation
    """
    def __init__(self, xml_path):
        if xml_path:
            self.load_model(xml_path)
            self.get_joint_limits()

        self.home_pose = np.array([
            0.0,
            -0.785,   # -45°
            0.0,
            -2.356,   # -135°
            0.0,
            1.571,   # 90°
            0.785    # 45°
        ])

    def load_model(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.003
        print(f"timestep: {self.model.opt.timestep}")
        self.data = mujoco.MjData(self.model)
        self.n_arm_dofs = 7
        self.n_gripper_dofs = 2
        self.n_dofs = self.model.nv 
        self.n_jnts = self.model.njnt
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]

    # set control torques for arm (actuator command)
    def set_arm_ctrl(self, q_arm):
        self.data.ctrl[:7] = q_arm

    # set control for gripper (actuator command)
    def set_gripper_ctrl(self, width):       
        self.data.ctrl[7] = width
        return True

    def get_joint_limits(self):
        self.arm_lower = self.model.jnt_range[:7, 0]
        self.arm_upper = self.model.jnt_range[:7, 1]
        
        self.gripper_lower = self.model.jnt_range[7:9, 0]
        self.gripper_upper = self.model.jnt_range[7:9, 1]
