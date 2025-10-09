from robot.pin_robot import PinRobot
import numpy as np
from scipy.optimize import least_squares


class IKSolver:
    """
    computes the inverse kinematics of the cartesian position using Least Squares "trf":
    as the panda is a 7dof robot and task space is 6dof, the solutions can be different
    """
    def __init__(self, robot: PinRobot, iterations=1000, tol=1e-10):
        self.robot = robot
        self.iterations = iterations
        self.tol = tol

    def solve(self, q_init, target_SE3):
       
        def residual(q):
            # Compute 6D error (rotation + translation)
            frame_error = self.robot.frame_error(q, target_SE3)

            # Split orientation and position errors
            ori_weight = 1.0
            pos_weight = 10.0 #0.01 precision
            ori_err = frame_error[:3] * ori_weight
            pos_err = frame_error[3:] * pos_weight

            return np.concatenate([ori_err, pos_err])

        result = least_squares(
            residual,
            q_init,
            xtol=self.tol,
            ftol=self.tol,
            gtol=self.tol,
            method="trf",
            bounds=(self.robot.lower, self.robot.upper),
            max_nfev=self.iterations
        )
        q_sol = result.x
        final_err = residual(q_sol)


        print(f"\nFinal joint configuration:{q_sol}")
        print(f"\nFinal end-effector error:{final_err}")

        return q_sol
