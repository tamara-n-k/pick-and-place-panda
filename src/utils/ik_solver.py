from robot.pin_robot import PinRobot
import numpy as np
from scipy.optimize import least_squares, minimize
import pinocchio as pin


class IKSolver:
    """
    computes the inverse kinematics of the cartesian position using Least Squares "trf":
    as the panda is a 7dof robot and task space is 6dof, the solutions can be different
    """
    def __init__(self, robot: PinRobot, margin_coll=0.02, beta_coll=10.0):
        self.robot = robot
        self.margin_coll = margin_coll
        self.beta_coll = beta_coll

    def solve(self, q_init, target_SE3, margin=0.05, weight_coll=100.0):
        def objective(q):
            # sqared frame error for scalar optimization
            err_vec = self.robot.frame_error(q, target_SE3)
            cost_pose = np.sum(err_vec**2)

            in_collision = self.robot.check_collisions(q) # returns True if any collision is detected
            # add a collision cost if in collision else 0
            cost_coll = weight_coll if in_collision else 0.0

            return cost_pose + cost_coll
        
        result = minimize(
            objective,
            q_init,
            method='SLSQP',
            bounds=list(zip(self.robot.lower, self.robot.upper)),
            options={'ftol': 1e-6, 'maxiter': 200}
        )

        return result.x if result.success else None