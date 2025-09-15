import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve


class IKPlanner:
    """
    computes the inverse kinematics of the cartesian position using Levenberg-Marquardt:
    as the panda is a 7dof robot and task space is 6dof, the solutions can be different
    """
    def __init__(self, environment, damping=1e-4, iterations=1000, tol=1e-4):
        self.env = environment
        self.damping = damping
        self.iterations = iterations
        self.tol = tol

    def solve(self, q_init, target_SE3):
        model = self.env.pin.model
        data = self.env.pin.data
        ee_frame_id = self.env.pin.ee_frame_id
        q = np.array(q_init, dtype=float)
        DT = 0.05 #integration step, reduce for smoothness


        for i in range(self.iterations):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, ee_frame_id)
            #pose error
            dMi = target_SE3.actInv(data.oMf[ee_frame_id])
            err = pin.log(dMi).vector#pin.log returns error in local end-effector frame

            if norm(err) < self.tol:
                print("Convergence achieved!")
                break
            if i >= self.iterations:
                print("Warning: maximum iterations reached, solution may not be accurate.")
                break
            #solver Levenberg-Marquardt
            J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
            # #add weight matrix to correct orientation
            W = np.diag([3.0, 3.0, 3.0,   # orientation weights
                        1.0, 1.0, 1.0])  # position weights

            JJ = J.T @ W @ J + self.damping * np.eye(model.nv)
            v = -np.linalg.solve(JJ, J.T @ W @ err)
            q = pin.integrate(model, q, v * DT)

        print(f"\nFinal joint configuration:{q}")
        print(f"\nFinal end-effector error:{err}")

        return q
