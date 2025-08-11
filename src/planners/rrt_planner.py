import mujoco
from ompl import base as ob
from ompl import geometric as og


class RRTPlanner:
    def __init__(self, environment):
        self.env = environment
        robot = self.env.robot

        self.bounds = ob.RealVectorBounds(robot.n_dofs)#robot's limits
        self.space = ob.RealVectorStateSpace(robot.n_dofs)#configuration space
        self.set_bounds(self.env.robot.lower_limits, self.env.robot.upper_limits)

        self.start = None
        self.goal = None

    def set_bounds(self, lower_limits, upper_limits):
        for i, low in enumerate(lower_limits):
            self.bounds.setLow(i, low)

        for i, high in enumerate(upper_limits):
            self.bounds.setHigh(i, high)

        self.space.setBounds(self.bounds)

    def define_start_goal_positions(self, start, goal):
        robot = self.env.robot
        assert len(start) == robot.n_dofs, "Start configuration does not match DoF"
        assert len(goal) == robot.n_dofs, "Goal configuration does not match DoF"
        
        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)

        for i in range(robot.n_dofs):
            self.start[i] = start[i]
            self.goal[i] = goal[i]

        #collision detection
    def is_valid(self, state):
        robot = self.env.robot
        for i in range(robot.n_dofs):
            robot.data.qpos[i] = state[i]

        mujoco.mj_forward(robot.model, robot.data)
        return robot.data.ncon == 0  # No contacts = collision-free

    def find_path(self, start = None, goal = None):
        #ompl planning
        if start is not None and goal is not None:
            self.define_start_goal_positions(start, goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")
        
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
            return path
        else:
            print("No path found")
            return None