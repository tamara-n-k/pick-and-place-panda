import numpy as np
import pinocchio as pin

class GraspSampler:
    def __init__(self, env, ik_solver):
        self.env = env
        self.robot = env.robot
        self.ik = ik_solver

    def get_valid_grasp(self, body_name, q_start, num_samples=16):
        plant_world_pos = self.env.get_body_pos(body_name) 
        robot_base_pos = self.env.robot.data.body("link0").xpos.copy()
        relative_pos = (plant_world_pos - robot_base_pos).astype(np.float64)
        
        grasp_pos = relative_pos + np.array([0, 0, 0.02])

        for angle in np.linspace(0, 2 * np.pi, num_samples):
            target_SE3 = pin.SE3.Identity()
            target_SE3.translation = grasp_pos
            
            R_down = pin.utils.rotate('x', np.pi)
            R_yaw = pin.utils.rotate('z', angle)
            target_SE3.rotation = R_yaw @ R_down

            q_goal = self.ik.solve(q_start, target_SE3)

            if q_goal is not None:
                if not self.env.pin.check_collisions(q_goal):
                    return q_goal, target_SE3
        
        return None, None