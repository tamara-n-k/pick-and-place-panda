import numpy as np
import pinocchio as pin

from utils.ik_solver import IKSolver

class GraspSampler:
    def __init__(self, solver: IKSolver):
        self.solver = solver
        # create standard seed for IK with elbow up
        self.default_seed = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    def _create_target_se3(self, pos, approach_dir, roll):
        pos = np.array(pos, dtype=np.float64).flatten()
        
        # z_axis pointing at plant
        z_axis = approach_dir / np.linalg.norm(approach_dir)
        
        # use cross product to find x axis perpendicular to world up and approach direction
        world_up = np.array([0, 0, 1], dtype=np.float64)
        x_axis = np.cross(world_up, z_axis)
        
        # perfectly vertical approach will cause cross product to be zero
        if np.linalg.norm(x_axis) < 1e-4:
            x_axis = np.array([1, 0, 0], dtype=np.float64)
        else:
            x_axis /= np.linalg.norm(x_axis)
            
        y_axis = np.cross(z_axis, x_axis)
        
        # rotation matrix with z-axis pointing at plant
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        # Apply roll around 
        R_final = R @ pin.exp3(np.array([0, 0, 1]) * roll)
        
        return pin.SE3(R_final.astype(np.float64), pos)

    def find_valid_grasp(self, plant_pos):
        """
        finds a  valid grasp configuration:
        tilt: how much the arm leans in
        azimuth: where the arm stands around the plant (front, left, back, right)
        roll: how the fingers are rotated around the approach axis

        """
        # 0 is vertical
        tilts = [0.0, 0.3, 0.6, 0.9] 
        azimuths = np.linspace(0, 2 * np.pi, 8) 
        rolls = [0, np.pi/4, -np.pi/4]

        for tilt in tilts:
            for az in azimuths:
                # convert tilt and azimuth to 3D vector that points from gripper to plant
                direction = np.array([
                    np.sin(tilt) * np.cos(az),
                    np.sin(tilt) * np.sin(az),
                    -np.cos(tilt)
                ])

                for roll in rolls:
                    # build the target SE3 with gripper2plant direction and roll
                    target_se3 = self._create_target_se3(plant_pos, direction, roll)
                    
                    q_sol = self.solver.solve(self.default_seed, target_se3)
                    
                    if q_sol is not None:
                        return q_sol, target_se3
        return None, None