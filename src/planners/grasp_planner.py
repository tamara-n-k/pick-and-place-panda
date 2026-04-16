import numpy as np
import pinocchio as pin

class GraspSampler:
    def __init__(self, env, ik_solver, rrt_planner):
        self.env = env
        self.robot = env.robot
        self.ik = ik_solver
        self.rrt = rrt_planner


    def get_valid_grasp(self, body_name, q_start, num_samples=16):
        plant_world_pos = self.env.get_body_pos(body_name) # get world position of the plant
        grasp_pos = plant_world_pos + np.array([0, 0, 0.06])
        tilts = [np.pi/6, np.pi/4, np.pi/3] # 30, 45, 60 degrees
        yaws = np.linspace(0, 2 * np.pi, num_samples)

        for tilt in tilts:
            for angle in yaws:
                target_SE3 = pin.SE3.Identity()
                target_SE3.translation = grasp_pos
            
                R_base = pin.utils.rotate('x', np.pi) # base rotation to face downwards
                R_tilt = pin.utils.rotate('y', tilt) # tilt rotation
                R_yaw = pin.utils.rotate('z', angle) # yaw rotation around vertical axis
                target_SE3.rotation = R_yaw @ R_tilt @ R_base
                q_goal = self.ik.solve(q_start, target_SE3)

            if q_goal is not None:
                if not self.rrt.collision_free(q_goal):
                    return q_goal, target_SE3
        
        return None, None