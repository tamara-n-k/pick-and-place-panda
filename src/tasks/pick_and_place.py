import mujoco
import numpy as np
import pinocchio as pin
import time
from planners.grasp_planner import GraspSampler
from fsm.pick_fsm import PickStateMachine  # Import our new state engine

class PickAndPlace:
    def __init__(self, env, planner, ik_solver):
        self.env = env
        self.planner = planner
        self.ik = ik_solver
        self.grasp_sampler = GraspSampler(self.env, self.ik, self.planner)
        self.approach_margin = 0.1

    def execute_move(self, q_goal_arm, plant_name, linear=False):
        current_q_full = self.env.get_state()
        q_start_arm = current_q_full[:7]
        q_gripper = current_q_full[7:]
        
        if not self.planner.collision_free(current_q_full):
            print("CRITICAL: Planner reports initialization state is in structural collision!")
        
        path_list_7dof = []
        self.planner.set_collision_mask(plant_name, ignore=True)

        if linear:
            print(f"Executing linear path to: {q_goal_arm}")
            path_arm = self.planner.find_linear_path(q_start_arm, q_goal_arm, plant_name, path_list_7dof)
            if path_arm is None:
                return False
        else:
            print(f"Executing RRT path to: {q_goal_arm}")
            path_arm = self.planner.find_path_rrt(q_start_arm, q_goal_arm, plant_name)
            if path_arm is None:
                return False
            
            pdef = self.planner.pdef
            if pdef.hasExactSolution():
                ompl_states = path_arm.getStates()
                path_list_7dof = [np.array([ompl_states[i][j] for j in range(7)]) for i in range(len(ompl_states))]
                    
        self.planner.set_collision_mask(plant_name, ignore=False)

        if len(path_list_7dof) > 0:
            full_path = [np.concatenate((q_arm, q_gripper)) for q_arm in path_list_7dof]
            
            # Step through points sequentially 
            for q in full_path:
                self.env.set_state(q)
                self.env.step()
                if self.env.mj_viewer.is_running():
                    time.sleep(0.01)
                    self.env.mj_viewer.sync()
            return True
        
        return False

    def pick_plant(self, plant_name):
        current_q = self.env.get_state()
        print(f"--- Sampling valid grasp for {plant_name} ---")
        
        q_grasp, grasp_SE3 = self.grasp_sampler.find_valid_grasp(
            plant_pos=self.env.get_body_pos(plant_name), 
            plant_name=plant_name
        )
        if q_grasp is None:
            print(f"No valid grasp found for {plant_name}")
            return False
            
        print(f"Found valid grasp for {plant_name} at position {grasp_SE3.translation}")
        
        # Calculate Approach & Lift Configurations
        approach_pos = grasp_SE3.translation + np.array([0, 0, 0.15]) 
        approach_SE3 = pin.SE3(grasp_SE3.rotation, approach_pos)
        
        q_approach = self.ik.solve(current_q[:7], approach_SE3)
        q_lift = self.ik.solve(q_init=q_grasp[:7], target_SE3=approach_SE3)
        
        if q_approach is None or q_lift is None:
            print("Aborting pipeline: IK configurations are mathematically unresolvable.")
            return False

        # Instantiate and delegate tracking to the State Machine
        fsm = PickStateMachine(self, plant_name, q_grasp, q_approach, q_lift)
        return fsm.run()