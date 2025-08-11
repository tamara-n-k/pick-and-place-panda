from robot.robot import Robot
from planners.rrt_planner import RRTPlanner
from environments.viewer import Viewer
if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/pick_place_scene.xml"
    robot = Robot(xml_path)

    planner = RRTPlanner(robot)
    start = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.04, 0.04]
    goal = [0.5, 0.0, 0.0, -1.0, 0.0, 1.2, -0.5, 0.0, 0.0]
    planner.define_start_goal_positions(start, goal)
    path = planner.find_path()
    Viewer.start_viewer(robot, path)
