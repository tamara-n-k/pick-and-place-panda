import mujoco

class Robot:
    def __init__(self, xml_path):
        if xml_path:
            self.load_model(xml_path)
            self.get_joint_limits()

    def load_model(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_dofs = self.model.nv 
        self.n_jnts = self.model.njnt
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]

    def get_joint_limits(self):
        lower_limits = []
        upper_limits = []

        for i in range(self.n_jnts):
            joint_type = self.model.jnt_type[i]
            
            if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                lower, upper = self.model.jnt_range[i]
                lower_limits.append(lower)
                upper_limits.append(upper)

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
