import mujoco
import mujoco.viewer
from ompl import base as ob
from ompl import geometric as og
import numpy as np
import time

class Robot:
    def __init__(self, xml_path):
        if xml_path:
            self.load_model(xml_path)
            self.get_joint_limits()

    def load_model(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_dofs = self.model.nv 
        self.n_jnts = self.model.njnt
        # joint_names = [model.joint(i).name for i in range(model.njnt)]

    def get_joint_limits(self):
        lower_limits = []
        upper_limits = []

        for i in range(self.n_jnts):
            joint_type = self.model.jnt_type[i]
            
            if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                lower, upper = self.model.jnt_range[i]
                lower_limits.append(lower)
                upper_limits.append(upper)

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

class Planner:
    def __init__(self, robot: Robot):
        self.bounds = ob.RealVectorBounds(robot.n_dofs)#robot's limits
        self.space = ob.RealVectorStateSpace(robot.n_dofs)#configuration space
        self.robot = robot
        self.set_bounds(robot.lower_limits, robot.upper_limits)

    def set_bounds(self, lower_limits, upper_limits):
        for i, low in enumerate(lower_limits):
            self.bounds.setLow(i, low)

        for i, high in enumerate(upper_limits):
            self.bounds.setHigh(i, high)

        self.space.setBounds(self.bounds)

    def define_start_goal_positions(self, start, goal):
        assert len(start) == self.robot.n_dofs, "Start configuration does not match DoF"
        assert len(goal) == self.robot.n_dofs, "Goal configuration does not match DoF"
        
        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)

        for i in range(self.robot.n_dofs):
            self.start[i] = start[i]
            self.goal[i] = goal[i]

        #collision detection
    def is_valid(self, state):
        for i in range(self.robot.n_dofs):
            self.robot.data.qpos[i] = state[i]

        mujoco.mj_forward(robot.model, robot.data)
        return robot.data.ncon == 0  # No contacts = collision-free

    def find_path(self):
        #ompl planning
        si = ob.SpaceInformation(self.space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))
        si.setup()

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(self.start, self.goal)

        planner = og.RRT(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        if planner.solve(5.0):#find solution within 5 seconds
            path = pdef.getSolutionPath()
            path.interpolate()  # smoother
            print("Found path with", path.getStateCount(), "states")
        else:
            print("No path found")
        return path


class Viewer:
    @staticmethod
    def start_viewer(robot: Robot, path):
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            for i in range(path.getStateCount()):
                state = path.getState(i)
                for j in range(robot.n_dofs):
                    robot.data.qpos[j] = state[j]#passes the joint position of the planner to the model
                mujoco.mj_forward(robot.model, robot.data)
                viewer.sync()
                time.sleep(0.01)

            # Keep window open
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)



if __name__ == "__main__":

    xml_path = "model/franka_emika_panda/pick_place_scene.xml"
    robot = Robot(xml_path)

    planner = Planner(robot)
    start = [0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.04, 0.04]
    goal = [0.5, 0.0, 0.0, -1.0, 0.0, 1.2, -0.5, 0.0, 0.0]
    planner.define_start_goal_positions(start, goal)
    path = planner.find_path()
    Viewer.start_viewer(robot, path)
