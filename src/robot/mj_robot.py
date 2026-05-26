import mujoco

class MJRobot:
    """
    holds the robot model data for the mujoco simulation
    """
    def __init__(self, xml_path):
        if xml_path:
            self.load_model(xml_path)
            self.get_joint_limits()

    def load_model(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.008
        print(f"timestep: {self.model.opt.timestep}")
        self.data = mujoco.MjData(self.model)
        self.n_arm_dofs = 7
        self.n_gripper_dofs = 2
        self.n_dofs = self.model.nv 
        self.n_jnts = self.model.njnt
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]


    def set_gripper_ctrl(self, width):       
        print(f"Setting gripper control to width: {width}")
        self.data.ctrl[7] = width
        return True

    def get_joint_limits(self):
        self.arm_lower = self.model.jnt_range[:7, 0]
        self.arm_upper = self.model.jnt_range[:7, 1]
        
        self.gripper_lower = self.model.jnt_range[7:9, 0]
        self.gripper_upper = self.model.jnt_range[7:9, 1]
