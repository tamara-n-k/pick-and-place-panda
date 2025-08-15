import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve

urdf_path = "model/franka_emika_panda/panda.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
#end effector
JOINT_ID = 7
ee_frame_name = next(f.name for f in model.frames if f.parentJoint == JOINT_ID)
ee_frame_id   = model.getFrameId(ee_frame_name)
#desired pose
oMdes = pin.SE3(np.eye(3), np.array([0., 1., 0.]))
#initial configuration
q = pin.neutral(model)
#computation parameters
eps    = 1e-4        # convergence threshold
IT_MAX = 1000        # max iterations
DT     = 0.1         # integration step
damp   = 1e-4        # Levenberg-Marquardt damping
# iterative IK loop
i = 0
while True:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, ee_frame_id)
    #pose error
    dMi = oMdes.actInv(data.oMf[ee_frame_id])
    err = pin.log(dMi).vector

    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break

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
