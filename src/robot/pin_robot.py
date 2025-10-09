import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class PinRobot:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        ee_frame_name = "hand"
        self.ee_frame_id   = self.model.getFrameId(ee_frame_name)
        self.lower =self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit

    def forward_kinematics(self, q):
        """
        computes the end-effector SE3 pose for the given joint configuration q
        """
        q = np.array(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id] # pin.SE3 format
    
    def frame_error(self, q, target_SE3):
        current = self.forward_kinematics(q)
        return pin.log(current.inverse() * target_SE3).vector
