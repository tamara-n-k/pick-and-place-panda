from planners.rrt_planner import RRTPlanner
from environments.environment import Environment
from utils.ik_solver import IKSolver
from tasks.pick_and_place import PickAndPlace

import time

if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/scene_real_floats.xml"
    urdf_path = "model/franka_emika_panda/panda.urdf"
    mesh_dir = "model/franka_emika_panda/assets"
    
    env = Environment(xml_path, urdf_path, mesh_dir)
    task = PickAndPlace(env, RRTPlanner(env), IKSolver(env.pin))
    plants = ["plant_5"]

    for plant in plants:
        task.pick_plant(plant)
    
    print("Task completed.")
    try:
        while env.mj_viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        pass

    


