import pinocchio as pin
import numpy as np
from scipy.optimize import least_squares


class IKPlanner:
    """
    computes the inverse kinematics of the cartesian position using Least Squares "trf":
    as the panda is a 7dof robot and task space is 6dof, the solutions can be different
    """
    def __init__(self, environment, iterations=1000, tol=1e-10):
        self.env = environment
        self.iterations = iterations
        self.tol = tol

    def solve(self, q_init, target_SE3):
        model = self.env.pin.model
        data = self.env.pin.data
        ee_frame_id = self.env.pin.ee_frame_id
        lower = model.lowerPositionLimit
        upper = model.upperPositionLimit

        
        def residual(q):
            q = np.array(q)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, ee_frame_id)


            # Compute 6D error (rotation + translation)
            dMi = target_SE3.actInv(data.oMf[ee_frame_id])
            err6 = pin.log(dMi).vector  # [rx, ry, rz, dx, dy, dz]

            # Split orientation and position errors
            ori_weight = 1.0
            pos_weight = 100.0 #0.01 precision
            ori_err = err6[:3] * ori_weight
            pos_err = err6[3:] * pos_weight

            return np.concatenate([ori_err, pos_err])

        result = least_squares(
            residual,
            q_init,
            xtol=self.tol,
            ftol=self.tol,
            gtol=self.tol,
            method="trf",
            bounds=(lower, upper),
            max_nfev=self.iterations
        )
        q_sol = result.x
        final_err = residual(q_sol)


        print(f"\nFinal joint configuration:{q_sol}")
        print(f"\nFinal end-effector error:{final_err}")

        return q_sol
