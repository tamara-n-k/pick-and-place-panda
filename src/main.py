from planners.rrt_planner import RRTPlanner
from environments.environment import Environment

if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/pick_place_scene.xml"

    start = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.04, 0.04]
    goal = [0.5, 0.0, 0.0, -1.0, 0.0, 1.2, -0.5, 0.0, 0.0]
    
    env = Environment(xml_path)
    env.set_goal(goal)
    planner = RRTPlanner(env)
    planned_path = planner.find_path(start, goal)
    if planned_path:
        env.run_viewer(planned_path)
