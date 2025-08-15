import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve


class IKPlanner:
    def __init__(self, environment, damping=1e-4, iterations=1000, tol=1e-4):
        self.env = environment
        self.damping = damping
        self.iterations = iterations
        self.tol = tol

    def solve(self, q_init_full, target_SE3):
        model = self.env.pin.model
        data = self.env.pin.data
        ee_frame_id = self.env.pin.ee_frame_id
        q = np.array(q_init_full[:-2], dtype=float)
        i = 0
        while True:
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, ee_frame_id)
            #pose error
            dMi = target_SE3.actInv(data.oMf[ee_frame_id])
            err = pin.log(dMi).vector

            if norm(err) < self.tol:
                success = True
                break
            if i >= self.iterations:
                success = False
                break
            DT = 0.1 #integration step
            J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)    #create solver Levenberg-Marquardt
            lambda_damp = 1e-4
            JJ = J @ J.T + lambda_damp * np.eye(6)
            v = -J.T @ np.linalg.solve(JJ, err)
            q = pin.integrate(model, q, v*DT)

            # if i % 10 == 0:
                    # print(f"{i}: error norm = {norm(err):.6f}")

            i += 1

        if success:
            print("Convergence achieved!")
        else:
            print("Warning: maximum iterations reached, solution may not be accurate.")

        print("\nFinal joint configuration:")
        print(q)
        print("\nFinal end-effector error:")
        print(err)
        return q
