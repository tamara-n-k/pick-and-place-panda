

import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = "model/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Print joint and actuator names for reference
print("Joints:", [model.joint(i).name for i in range(model.njnt)])
print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

def joint_trajectory(t):
    return np.array([
        0.5 * np.sin(2 * np.pi * 0.2 * t),  # joint 0
        0, 0, 0, 0, 0, 0, 0  # others stay fixed
    ])


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        t = time.time() - start  # real time

        target_pos = joint_trajectory(t)

        if model.nu != 8:
            raise ValueError(f"Expected 8 actuators, got {model.nu}")
        
        data.ctrl[:] = target_pos

        mujoco.mj_step(model, data)
        viewer.sync()
        



