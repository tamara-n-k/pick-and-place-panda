import numpy as np
import pinocchio as pin


class PickAndPlace:
    def __init__(self, env, planner, ik_solver):
        self.env = env
        self.planner = planner
        self.ik = ik_solver
        self.approach_margin = 0.3
        
    def get_grasp_point(self, body_name):
        plant_world_pos = self.env.get_body_pos(body_name) 
        robot_base_pos = self.env.robot.data.body("link0").xpos.copy()
        
        # Calculate position relative to the robot base
        relative_pos = plant_world_pos - robot_base_pos
        grasp_pos = relative_pos + np.array([0, 0, 0.02])
        
        return grasp_pos

    def execute_move(self, target_pos):
        if isinstance(target_pos, str):
            target_pos = self.get_grasp_point(target_pos)
        target_SE3 = pin.SE3.Identity()
        target_SE3.translation = target_pos
        target_SE3.rotation = pin.utils.rotate('x', np.pi) # face downwards
        
        full_q_start = self.env.get_state()
        q_start_arm = full_q_start[:7] # extract arm joints, ignore gripper state
        q_goal_arm = self.ik.solve(q_start_arm, target_SE3)
        
        if q_goal_arm is None:
            print(f"IK failed for position {target_pos}")
            return False
        
        print(f"Planning move from {q_start_arm} to {q_goal_arm}")
        current_gripper = self.env.get_state()[7:] # gripper state remains unchanged during the move
        path_arm = self.planner.find_path(full_q_start, np.concatenate((q_goal_arm, current_gripper)))
        if path_arm:
            states = path_arm.getStates()
            
            current_gripper = self.env.get_state()[7:]
            
            # Convert each state to a numpy array and concatenate the gripper
            full_path = []
            for state in states:
                # OMPL states need to be converted to numpy/list
                q_step = np.array([state[i] for i in range(7)])
                full_path.append(q_step)
            
            self.env.viewer.run_trajectory(self.env, full_path, speed=0.5)
            return True
        print(f"No valid path found for target position: {target_pos}")
        return False

    def pick_plant(self, plant_name):
        grasp_pos = self.get_grasp_point(plant_name)
        # open gripper
        self.set_gripper(open=True)
        
        # approach
        print(f"--- Picking {plant_name} ---")
        approach_pos = grasp_pos + np.array([0, 0, self.approach_margin])
        self.execute_move(approach_pos)
        
        # descend
        self.execute_move(grasp_pos)
        
        # close Gripper
        self.set_gripper(open=False)
        
        # lift
        self.execute_move(approach_pos)

    def set_gripper(self, open=True):
        target = 0.04 if open else 0.0
        self.env.robot.data.ctrl[7] = target
        for _ in range(50):
            self.env.step()