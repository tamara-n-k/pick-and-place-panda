import numpy as np
import pinocchio as pin

from environments.environment import Environment
from utils.ik_solver import IKSolver
from planners.motion_planner import MotionPlanner

class GraspSampler:
    def __init__(self, env: Environment, solver: IKSolver, planner: MotionPlanner):
        self.env = env
        self.solver = solver
        self.planner = planner

        # create standard seed for IK with elbow up
        self.default_seed = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    def _calculate_joint_penalty(self, q):
        q_limit_min = self.env.pin.lower
        q_limit_max = self.env.pin.upper
        q_center = (q_limit_min + q_limit_max) / 2.0
        
        return np.sum((q - q_center)**2)
    

    def _calculate_clearance(self):# binary clearance check for hand
        d = self.planner.planner_data
        m = self.env.robot.model

        for i in range(d.ncon):# number of collisions found
            con = d.contact[i]# contact information for the i-th collision
            g1 = m.geom(con.geom1).name or "" # name of the first geometry involved in the collision, or empty string if not found
            g2 = m.geom(con.geom2).name or ""

            # If the palm/hand is hitting anything, this is a bad grasp
            if "hand" in g1 or "hand" in g2:
                print(f"Collision detected with hand geometry: {g1} vs {g2}")
                return 0.0
                
        return 1.0 # clear


    def _create_target_se3(self, pos, approach_dir, roll):
        offset = 0.10 
        pos = np.array(pos, dtype=np.float64).flatten()
        safe_pos = pos - (approach_dir * offset)
        
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
        
        return pin.SE3(R_final.astype(np.float64), safe_pos)


    def find_valid_grasp(self, plant_pos):
        candidates = []
        tilts = [0.0, 0.3, 0.6] 
        azimuths = np.linspace(0, 2 * np.pi, 8) 
        rolls = [0, np.pi/4, -np.pi/4]

        for tilt in tilts:
            for az in azimuths:
                direction = np.array([
                    np.sin(tilt) * np.cos(az),
                    np.sin(tilt) * np.sin(az),
                    -np.cos(tilt)
                ])

                for roll in rolls:
                    target_se3 = self._create_target_se3(plant_pos, direction, roll)
                    q_sol = self.solver.solve(self.default_seed, target_se3)
                    
                    if q_sol is not None:
                        if not self.planner.collision_free(q_sol):
                            continue

                        # Calculate metrics
                        clearance = self._calculate_clearance()
                        joint_penalty = self._calculate_joint_penalty(q_sol)
                        
                        # Score: maximize clearance, minimize joint distance, favor top-down (low tilt)
                        score = (clearance * 10.0) - (joint_penalty * 1.0) - (tilt * 2.0)
                        
                        candidates.append({'q': q_sol, 'pose': target_se3, 'score': score})

        if not candidates:
            return None, None

        # Pick the winner
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]['q'], candidates[0]['pose']