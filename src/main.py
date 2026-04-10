from planners.rrt_planner import RRTPlanner
from environments.environment import Environment
from environments.viewer import Viewer
from utils.ik_solver import IKSolver


if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/scene_real_floats.xml"
    urdf_path = "model/franka_emika_panda/panda.urdf"
    mesh_dir = "model/franka_emika_panda/assets"
    #joint positions: ['universe', 'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    q_start = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7]

    #cube position
    goal_cartesian = [0.43, -0.3, 0.7]
    
    env = Environment(xml_path, urdf_path, mesh_dir)
    env.set_state(q_start)
    
    #set cartesian position
    env.set_goal_cartesian(goal_cartesian)
    ik = IKSolver(env.pin)
    q_goal = ik.solve(q_start, env.goal_cartesian)
    env.set_goal_joint(q_goal)

    #plan in joint space
    planner = RRTPlanner(env)
    planned_path = planner.find_path(env.get_state(), env.get_goal())
    if planned_path:
        Viewer.run_trajectory(env, planned_path, speed=0.5)

