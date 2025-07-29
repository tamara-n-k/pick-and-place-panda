import mujoco
import mujoco.viewer
from ompl import base as ob
from ompl import geometric as og
import numpy as np
import time

model_path = "model/franka_emika_panda/pick_place_scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

n_dofs = model.nv  # Number of DoFs
joint_names = [model.joint(i).name for i in range(model.njnt)]

lower_limits = []
upper_limits = []

for i in range(model.njnt):
    joint_type = model.jnt_type[i]
    
    # Only consider hinge or slide joints (not free joints etc.)
    if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        lower, upper = model.jnt_range[i]
        lower_limits.append(lower)
        upper_limits.append(upper)

print("Lower limits:", lower_limits)
print("Upper limits:", upper_limits)

bounds = ob.RealVectorBounds(n_dofs)
space = ob.RealVectorStateSpace(n_dofs)

for i, low in enumerate(lower_limits):
    bounds.setLow(i, low)

for i, high in enumerate(upper_limits):
    bounds.setHigh(i, high)


space.setBounds(bounds)


start = ob.State(space)
goal = ob.State(space)

start_vals = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.04, 0.04]
goal_vals = [0.5, 0.0, 0.0, -1.0, 0.0, 1.2, -0.5, 0.0, 0.0]

for i in range(n_dofs):
    start[i] = start_vals[i]
    goal[i] = goal_vals[i]

def is_valid(state):
    for i in range(n_dofs):
        data.qpos[i] = state[i]
    mujoco.mj_forward(model, data)

    return data.ncon == 0  # No contacts = collision-free

si = ob.SpaceInformation(space)
si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
si.setup()

pdef = ob.ProblemDefinition(si)
pdef.setStartAndGoalStates(start, goal)

planner = og.RRT(si)
planner.setProblemDefinition(pdef)
planner.setup()

if planner.solve(5.0):
    path = pdef.getSolutionPath()
    path.interpolate()  # smoother
    print("Found path with", path.getStateCount(), "states")
else:
    print("No path found")




with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(path.getStateCount()):
        state = path.getState(i)
        for j in range(n_dofs):
            data.qpos[j] = state[j]
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)

    # Keep window open
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)



