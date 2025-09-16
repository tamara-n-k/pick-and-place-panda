import pinocchio as pin
import numpy as np
from scipy.optimize import least_squares


class IKPlanner:
    """
    computes the inverse kinematics of the cartesian position using Least Squares "trf":
    as the panda is a 7dof robot and task space is 6dof, the solutions can be different
    """
    def __init__(self, environment, iterations=1000, tol=1e-4):
        self.env = environment
        self.iterations = iterations
        self.tol = tol

    def solve(self, q_init, target_SE3):
        model = self.env.pin.model
        data = self.env.pin.data
        ee_frame_id = self.env.pin.ee_frame_id

        def residual(q):
            q = np.array(q)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, ee_frame_id)
            dMi = target_SE3.actInv(data.oMf[ee_frame_id])
            err = pin.log(dMi).vector  # 6D error [angular, linear]
            return err

        result = least_squares(
            residual,
            q_init,
            xtol=self.tol,
            ftol=self.tol,
            max_nfev=self.iterations,
            method="trf"
        )
        q_sol = result.x
        final_err = residual(q_sol)


        print(f"\nFinal joint configuration:{q_sol}")
        print(f"\nFinal end-effector error:{final_err}")

        return q_sol
