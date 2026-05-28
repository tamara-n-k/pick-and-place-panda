from environments.viewer import Viewer
from planners.motion_planner import MotionPlanner
from environments.environment import Environment
from utils.ik_solver import IKSolver
from tasks.pick_and_place import PickAndPlace
import numpy as np

import time

if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/table_panda_scene.xml"
    urdf_path = "model/franka_emika_panda/panda.urdf"
    mesh_dir = "model/franka_emika_panda/assets"
    
    env = Environment(xml_path, urdf_path, mesh_dir)

    env.set_state(env.robot.home_pose)
    print("Home position reached: ", env.robot.data.qpos[:7])

    Viewer.setup_initial_view(env)

    # initial settling loop 
    for _ in range(100):
        env.robot.set_arm_ctrl(env.robot.home_pose)
        env.step()
        if env.mj_viewer.is_running():
            env.mj_viewer.sync()
        time.sleep(env.robot.model.opt.timestep)

    task = PickAndPlace(env, MotionPlanner(env), IKSolver(env.pin))
    plants = ["plant_1", "plant_2"]
    # hole offsets of the place down tray
    hole_offsets = [
        np.array([0.28, 0.10, -0.18]),
        np.array([0.45, 0.10, -0.18])
                ]

    for plant in plants:
        success = task.execute_pick_and_place(plant_name=plant, tray_name="tray_place_down", hole_offsets=hole_offsets)
        if not success:
            print(f"Task failed for {plant}.")
        else:
            print(f"Task succeeded for {plant}.")
    
    print("Task completed.")
    try:
        while env.mj_viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        pass

    

