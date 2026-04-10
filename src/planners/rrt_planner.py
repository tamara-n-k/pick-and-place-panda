import mujoco
from ompl import base as ob
from ompl import geometric as og


class RRTPlanner:
    """
    takes joint positions and plans trajectory from start point to end point
    """
    def __init__(self, environment):
        self.env = environment
        robot = self.env.robot

        self.bounds = ob.RealVectorBounds(robot.n_arm_dofs)#robot's limits
        self.space = ob.RealVectorStateSpace(robot.n_arm_dofs)#configuration space
        self.set_bounds(self.env.robot.arm_lower, self.env.robot.arm_upper)

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
        assert len(start) == robot.n_arm_dofs, "Start configuration does not match DoF"
        assert len(goal) == robot.n_arm_dofs, "Goal configuration does not match DoF"
        
        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)

        for i in range(robot.n_arm_dofs):
            self.start[i] = start[i]
            self.goal[i] = goal[i]

        #collision detection
    def collistion_free(self, state):
        robot = self.env.robot
        for i in range(robot.n_arm_dofs):
            robot.data.qpos[i] = state[i]
        mujoco.mj_forward(robot.model, robot.data)
        return robot.data.ncon <=2  # allow up to 2 contacts (gripper touching the plant)

    def find_path(self, start = None, goal = None):
        #ompl planning
        if start is not None and goal is not None:
            self.define_start_goal_positions(start, goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")
        
        si = ob.SpaceInformation(self.space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.collistion_free))
        si.setup()

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(self.start, self.goal)
        # planner = og.RRTConnect(si)  # much faster and often cleaner
        planner = og.RRTstar(si)
        planner.setRange(0.2)
        planner.setProblemDefinition(pdef)
        planner.setup()

        if planner.solve(2.0):#find solution within 2 seconds
            path = pdef.getSolutionPath()
            ps = og.PathSimplifier(si)
            ps.simplify(path, 1.0) # simplify the path with a time limit of 1 second
            path.interpolate(200) 
            print(f"Found path with {path.getStateCount()} states")
            return path
        else:
            print("No path found")
            return None