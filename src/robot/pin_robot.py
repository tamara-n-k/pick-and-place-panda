import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class PinRobot:
    def __init__(self, urdf_path: str, mesh_dir: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        ee_frame_name = "hand"
        self.ee_frame_id   = self.model.getFrameId(ee_frame_name)
        self.lower =self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        self.geom_model = pin.buildGeomFromUrdf(self.model, urdf_path, 
                                                pin.GeometryType.COLLISION,
                                                package_dirs=mesh_dir)
        
        self.geom_model.addAllCollisionPairs()
        
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        self.tcp_offset = pin.SE3(np.eye(3), np.array([0, 0, 0.103]))
        
    def update_state(self, q):
        q = np.array(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data)
        self.q = q

    def forward_kinematics(self, q, update_geom=False):
        q = np.array(q)
        # update joints
        pin.forwardKinematics(self.model, self.data, q)
        # update frames
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        if update_geom:
            pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data)
        hand_pose = self.data.oMf[self.ee_frame_id]
        tcp_pose = hand_pose * self.tcp_offset
        return tcp_pose
    
    def frame_error(self, q, target_SE3):
        current_tcp = self.forward_kinematics(q)
        return pin.log(current_tcp.inverse() * target_SE3).vector
    
    def check_collisions(self, q):# internal map using the kinematic model
        # Standard FK update
        pin.forwardKinematics(self.model, self.data, q)
        # Sync geometry shapes to the joints
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data)
        # Perform the check
        pin.computeCollisions(self.geom_model, self.geom_data, False)
        
        for res in self.geom_data.collisionResults:
            if res.isCollision():
                return True
        return False
