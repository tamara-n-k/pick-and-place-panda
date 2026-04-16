import numpy as np
import pinocchio as pin

from environments.viewer import Viewer
from planners.grasp_planner import GraspSampler

class PickAndPlace:
    def __init__(self, env, planner, ik_solver):
        self.env = env
        self.planner = planner
        self.ik = ik_solver
        self.grasp_sampler = GraspSampler(self.env, self.ik, self.planner)
        self.approach_margin = 0.3

    def execute_move(self, target_conf, plant_name):
        if isinstance(target_conf, np.ndarray) and target_conf.shape == (3,):
            target_pos = target_conf.astype(np.float64).flatten()
            target_SE3 = pin.SE3.Identity()
            target_SE3.translation = target_pos
            target_SE3.rotation = pin.utils.rotate('x', np.pi) # face downwards
            
            current_q = self.env.get_state()
            q_start_arm = current_q[:7] # extract arm joints, ignore gripper state
            q_gripper = current_q[7:] # gripper state
            q_goal_arm = self.ik.solve(q_start_arm, target_SE3)
        else:
            q_start_arm = self.env.get_state()[:7]
            q_goal_arm = target_conf[:7]
        
        if q_goal_arm is None:
            print(f"IK failed for position {target_conf}")
            return False
        
        print(f"Planning move from {q_start_arm} to {q_goal_arm}")
        path_arm = self.planner.find_path(q_start_arm, q_goal_arm, plant_name)
        if path_arm is None:
            print(f"Planning failed for target position: {target_conf}")
            return False
        
        pdef = self.planner.pdef

        if pdef.hasExactSolution():
            ompl_states = path_arm.getStates()
            path_list_7dof = []
            for i in range(len(ompl_states)):
                state_7dof = np.array([ompl_states[i][j] for j in range(7)]) # extract only the arm joints
                path_list_7dof.append(state_7dof)

            q_gripper = self.env.get_state()[7:] # current gripper state
            full_path = [np.concatenate((q_arm, q_gripper)) for q_arm in path_list_7dof]

            Viewer.run_trajectory(self.env, full_path, speed=0.5)
            self.env.set_state(full_path[-1])
            return True
        
        print(f"No valid path found for target position: {target_conf}")
        return False


    def pick_plant(self, plant_name):
        current_q = self.env.get_state()
        print(f"--- Sampling valid grasp for {plant_name} ---")
        q_grasp, grasp_SE3 = self.grasp_sampler.get_valid_grasp(plant_name, current_q[:7])
        if q_grasp is None:
            print(f"No valid grasp found for {plant_name}")
            return False
        print(f"Found valid grasp for {plant_name} at position {grasp_SE3.translation}")
        approach_SE3 = grasp_SE3.copy()
        approach_SE3.translation -= grasp_SE3.rotation @ np.array([0, 0, 0.2]) # approach from above
        q_approach = self.ik.solve(current_q[:7], approach_SE3)
        # open gripper
        if not self.env.robot.set_gripper_ctrl(width=0.04):
            print("Failed to open gripper.")
            return
        for _ in range(50):
            self.env.step()
        
        # approach
        print(f"--- Picking {plant_name} ---")
        
        if not self.execute_move(q_approach, plant_name):
            print("Failed to approach plant.")
            return
        print(f"--- Reached approach pose {approach_SE3.translation} for {plant_name} ---")
        
        # descend
        print(f"--- Descending to grasp {plant_name} ---")
        if not self.execute_move(q_grasp, plant_name):
            print("Failed to descend to grasp plant.")
            return

        # close Gripper
        print(f"--- Grasping {plant_name} ---")
        if not self.env.robot.set_gripper_ctrl(width=0.0):
            print("Failed to close gripper.")
            return
        for _ in range(50):
            self.env.step()

        # lift
        print(f"--- Lifting {plant_name} ---")
        if not self.execute_move(approach_SE3, plant_name):
            print("Failed to lift plant.")
            return
