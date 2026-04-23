import mujoco
from ompl import base as ob
from ompl import geometric as og

class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, collision_func):
        super().__init__(si)
        self.collision_func = collision_func

    def isValid(self, state):
        return self.collision_func(state)
    

class MotionPlanner:
    """
    takes joint positions and plans trajectory from start point to end point
    """
    def __init__(self, environment):
        self.env = environment
        robot = self.env.robot

        self.bounds = ob.RealVectorBounds(robot.n_arm_dofs)#robot's limits
        self.space = ob.RealVectorStateSpace(robot.n_arm_dofs)#configuration space
        self.set_bounds(self.env.robot.arm_lower, self.env.robot.arm_upper)

        # setup space information
        self.si = ob.SpaceInformation(self.space)
        self.svc = ValidityChecker(self.si, self.collision_free)
        self.si.setStateValidityChecker(self.svc)
        self.si.setup()
        # problem definition
        self.pdef = ob.ProblemDefinition(self.si)

        self.pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.si))
        self.planner = og.RRTConnect(self.si)
        self.planner.setRange(0.2) # Step size
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()

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
        
        self.start = self.space.allocState()
        self.goal = self.space.allocState()

        for i in range(robot.n_arm_dofs):
            self.start[i] = start[i]
            self.goal[i] = goal[i]

        #collision detection
    def collision_free(self, state):
        robot = self.env.robot
        for i in range(robot.n_arm_dofs):
            robot.data.qpos[i] = state[i]

        mujoco.mj_forward(robot.model, robot.data)
        return robot.data.ncon <=2  # allow up to 2 contacts (gripper touching the plant)

    def set_collision_mask(self, body_name, ignore=True):
        collision_name = f"{body_name}_cube"
        geom_object = self.env.robot.model.geom(collision_name)
        target_geom_id = geom_object.id
        if ignore:
            self.env.robot.model.geom_contype[target_geom_id] = 0
            self.env.robot.model.geom_conaffinity[target_geom_id] = 0
            self.env.robot.model.geom_contype[self.env.robot.model.geom("hand_capsule").id] = 0
        else:
            self.env.robot.model.geom_contype[target_geom_id] = 2
            self.env.robot.model.geom_conaffinity[target_geom_id] = 2


    def _validate_states(self, plant_name):
        start_valid = self.si.isValid(self.start)
        goal_valid = self.si.isValid(self.goal)

        if not start_valid or not goal_valid:
            if not start_valid: 
                print("CRITICAL: Start state in collision!")
            if not goal_valid: 
                print("CRITICAL: Goal state in collision!")
                mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
                
                print(f"Number of contacts: {self.env.robot.data.ncon}")
                for i in range(self.env.robot.data.ncon):
                    con = self.env.robot.data.contact[i]
                    geom1 = self.env.robot.model.geom(con.geom1).name
                    geom2 = self.env.robot.model.geom(con.geom2).name
                    print(f" CONTACT DETECTED: {geom1} <--> {geom2}")
            return False
        return True
    

    def find_path_rrt(self, start = None, goal = None, plant_name=None):
        #ompl planning
        if start is not None and goal is not None:
            self.define_start_goal_positions(start, goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")
        self.set_collision_mask(plant_name, ignore=True) # ensure plant is considered in collision checking

        if not self._validate_states(plant_name):
            self.set_collision_mask(plant_name, ignore=False)
            return None
            
        # Clear OMPL's internal state before planning
        self.pdef.clearSolutionPaths()
        self.pdef.clearStartStates()
        self.pdef.setStartAndGoalStates(self.start, self.goal)
        self.planner.clear()

        path = None
        if self.planner.solve(2.0):
            # only return the path if it actually reached the goal
            if self.pdef.hasExactSolution():
                path = self.pdef.getSolutionPath()
                
                ps = og.PathSimplifier(self.si)
                ps.simplify(path, 1.0)
                path.interpolate(200) 
                
                print(f"Found EXACT path with {path.getStateCount()} states")
                return path
            else:
                print("Found only an APPROXIMATE path (not safe to execute)")
                return None
        else:
            print("No path found at all")
            return None
        
        
    def find_linear_path(self, q_start, q_goal, plant_name, path_list):
        if q_start is not None and q_goal is not None:
            self.define_start_goal_positions(q_start, q_goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")
        self.set_collision_mask(plant_name, ignore=True) # ensure plant is considered in collision checking

        if not self._validate_states(plant_name):
            self.set_collision_mask(plant_name, ignore=False)
            return None
        
        steps = 100
        for i in range(steps):
            alpha = i / (steps - 1)
            q_interp = (1 - alpha) * q_start + alpha * q_goal
            if not self.collision_free(q_interp):
                print(f"Collision detected at step {i} of linear descent!")
                return None
                
            path_list.append(q_interp)
        return path_list