from planners.rrt_planner import RRTPlanner
from environments.environment import Environment
from planners.ik_planner import IKPlanner
import numpy as np

if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/pick_place_scene.xml"
    urdf_path = "model/franka_emika_panda/panda.urdf"
    #joint positions: ['universe', 'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    q_start = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.04, 0.04]
    q_goal = [0.5, 0.0, 0.0, -1.0, 0.0, 1.2, -0.5, 0.04, 0.04]
    
    goal_cartesian = [0, 0.45, 0.48]#this is where the cube is placed
    
    env = Environment(xml_path, urdf_path)
    env.set_state(q_start)
    #set joint position
    # env.set_goal_joint(q_goal)
    #OR
    #set cartesian position
    env.set_goal_cartesian(goal_cartesian)
    ik = IKPlanner(env)
    cart2q = ik.solve(env.get_state(), env.goal_cartesian)
    full_goal = np.concatenate([cart2q, q_goal[-2:]], dtype=float)
    print("full_goal", full_goal)
    env.set_goal_joint(full_goal)

    planner = RRTPlanner(env)
    planned_path = planner.find_path(env.get_state(), full_goal)
    if planned_path:
        env.run_viewer(planned_path)

