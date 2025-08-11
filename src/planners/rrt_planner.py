import mujoco
from ompl import base as ob
from ompl import geometric as og

from robot.robot import Robot

class RRTPlanner:
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

        mujoco.mj_forward(self.robot.model, self.robot.data)
        return self.robot.data.ncon == 0  # No contacts = collision-free

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