import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pathlib import Path

class PinRobot:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        print("Loaded URDF model", self.model.name)
        print(self.model.nq)   # should be 9
        #end effector
        #TODO set last joint automatically
        ee_frame_name = next(f.name for f in self.model.frames if f.parentJoint == 7)
        self.ee_frame_id   = self.model.getFrameId(ee_frame_name)

