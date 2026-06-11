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
        # Separate MuJoCo data used only for planning queries.
        # This lets collision checks run FK/contact detection without moving the live simulation.
        self.planner_data = mujoco.MjData(self.env.robot.model)
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

    # go through the MuJoCo body tree to see if the collision object is part of the robot
    # all robot links are descendants of link0
    def _body_is_robot(self, body_id):
        m = self.env.robot.model
        robot_root_id = m.body("link0").id

        while body_id != 0:
            if body_id == robot_root_id:
                return True
            body_id = m.body_parentid[body_id]
        return False
    
    # the robot is not allowed to collide with those objects during motion planning
    def _body_is_forbidden_object(self, body_id):
        m = self.env.robot.model
        body_name = m.body(body_id).name or ""
        forbidden_objects = ["table", "tray", "plant"]

        return any(obj in body_name for obj in forbidden_objects)


    # the contact is invalid only when it is robot and obstacle collision
    # contacts between non-robot objects are ignored by the motion planner
    def _contact_is_forbidden(self, con):
        m = self.env.robot.model
        geom1 = m.geom(con.geom1)
        geom2 = m.geom(con.geom2)

        side1_is_robot = self._body_is_robot(geom1.bodyid)
        side2_is_robot = self._body_is_robot(geom2.bodyid)
        side1_is_forbidden = self._body_is_forbidden_object(geom1.bodyid)
        side2_is_forbidden = self._body_is_forbidden_object(geom2.bodyid)

        return (
            (side1_is_robot and side2_is_forbidden)
            or (side2_is_robot and side1_is_forbidden)
        )

    def _forbidden_contact_names(self):
        d = self.planner_data
        m = self.env.robot.model
        contacts = []

        for i in range(d.ncon):
            con = d.contact[i]
            if not self._contact_is_forbidden(con):
                continue

            g1 = m.geom(con.geom1).name or "unknown" or ""
            g2 = m.geom(con.geom2).name or "unknown" or ""
            b1 = m.body(m.geom(con.geom1).bodyid).name or "unknown" or ""
            b2 = m.body(m.geom(con.geom2).bodyid).name or "unknown" or ""
            contacts.append((g1, b1, g2, b2))

        return contacts


    def collision_free(self, state):
        d = self.planner_data
        m = self.env.robot.model
        for i in range(7):
            d.qpos[i] = state[i]
        mujoco.mj_forward(m, d) # run forward kinematics without affecting viewer

        for i in range(d.ncon):
            if self._contact_is_forbidden(d.contact[i]):
                return False

        return True


    def set_collision_mask(self, target_body_name, ignore=True):
        robot_part_names = ["hand", "left_finger", "right_finger"]
        try:
            target_body_id = self.env.robot.model.body(target_body_name).id
        except:
            target_body_id = self.env.robot.model.geom(target_body_name).bodyid

        for i in range(self.env.robot.model.ngeom):
            geom_name = self.env.robot.model.geom(i).name
            body_id = self.env.robot.model.geom(i).bodyid
            body_name = self.env.robot.model.body(body_id).name

            if any(part in body_name for part in robot_part_names) or body_id == target_body_id:
                if ignore:
                    self.env.robot.model.geom_contype[i] = 0
                    self.env.robot.model.geom_conaffinity[i] = 0
                else:
                    self.env.robot.model.geom_contype[i] = 1
                    self.env.robot.model.geom_conaffinity[i] = 1


    def _validate_states(self, plant_name):
        start_valid = self.si.isValid(self.start)
        goal_valid = self.si.isValid(self.goal)

        if not start_valid or not goal_valid:
            d = self.planner_data 
            m = self.env.robot.model
            print(f"CRITICAL: {'Start' if not start_valid else 'Goal'} state in collision!")
            for g1, b1, g2, b2 in self._forbidden_contact_names():
                print(f" CONTACT: {g1} ({b1}) <--> {g2} ({b2})")
                
            return False
        return True
    

    def find_path_rrt(self, start = None, goal = None, plant_name=None):
        #ompl planning
        self.set_collision_mask(plant_name, ignore=True) # Ignore plant contacts while planning with the object attached
        if start is not None and goal is not None:
            self.define_start_goal_positions(start, goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")

        if not self._validate_states(plant_name):
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
        
        
    def find_linear_path(self, q_start, q_goal, plant_name):
        path_list = []
        self.set_collision_mask(plant_name, ignore=True) # Ignore plant contacts while planning with the object attached
        if q_start is not None and q_goal is not None:
            self.define_start_goal_positions(q_start, q_goal)
        elif self.start is None or self.goal is None:
            raise ValueError("Start and goal states must be defined")

        if not self._validate_states(plant_name):
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
