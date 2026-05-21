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
    task = PickAndPlace(env, MotionPlanner(env), IKSolver(env.pin))
    plants = ["plant_5"]

    for plant in plants:
        task.execute_pick_and_place(plant_name=plant, tray_name="tray_place_down", hole_offset_local=np.array([0.28, 0.1, -0.18]))
    
    print("Task completed.")
    try:
        while env.mj_viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        pass

    


