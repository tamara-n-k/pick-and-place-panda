import mujoco
import numpy as np
import pinocchio as pin
import time
from planners.grasp_planner import GraspSampler
from fsm.pick_fsm import PickAndPlaceStateMachine

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

    def execute_pick_and_place(self, plant_name, tray_name, hole_offset_local):
        current_q = self.env.get_state()
        
        pick_configs = self._compute_pick_configurations(plant_name, current_q)
        if pick_configs is None:
            return False
        q_grasp, q_pick_approach, q_pick_lift = pick_configs

        place_configs = self._compute_place_configurations(tray_name, hole_offset_local, q_pick_lift)
        if place_configs is None:
            return False
        q_place_approach, q_place_drop = place_configs

        print("\n--- Initializing Pick and Place State Machine Engine ---")
        fsm = PickAndPlaceStateMachine(
            task_runner=self,
            plant_name=plant_name,
            tray_name=tray_name,
            q_pick_approach=q_pick_approach,
            q_grasp=q_grasp,
            q_pick_lift=q_pick_lift,
            q_place_approach=q_place_approach,
            q_place_drop=q_place_drop
        )
        return fsm.run()
    

    def _compute_pick_configurations(self, plant_name, current_q):
        print(f"\n--- Sampling valid grasp for {plant_name} ---")
        plant_pos = self.env.get_body_pos(plant_name)
        
        q_grasp, grasp_SE3 = self.grasp_sampler.find_valid_grasp(
            plant_pos=plant_pos, 
            plant_name=plant_name
        )
        if q_grasp is None:
            print(f"No valid pick grasp found for {plant_name}")
            return None
            
        pick_approach_pos = grasp_SE3.translation + np.array([0, 0, 0.15]) 
        pick_approach_SE3 = pin.SE3(grasp_SE3.rotation, pick_approach_pos)
        
        q_pick_approach = self.ik.solve(current_q[:7], pick_approach_SE3)
        q_pick_lift = self.ik.solve(q_init=q_grasp[:7], target_SE3=pick_approach_SE3)

        if q_pick_approach is None or q_pick_lift is None:
            print("IK failed for pick approach or lift configurations.")
            return None

        return q_grasp, q_pick_approach, q_pick_lift
    
    def _compute_place_configurations(self, tray_name, hole_offset_local, q_pick_lift):
        print(f"\n--- Calculating place targets for {tray_name} ---")
        
        tray_world_pos = self.env.robot.data.body(tray_name).xpos
        tray_world_mat = self.env.robot.data.body(tray_name).xmat.reshape(3, 3)
        hole_world_pos = tray_world_pos + (tray_world_mat @ hole_offset_local)

        straight_down_dir = np.array([0.0, 0.0, -1.0])
        place_world_SE3 = self.grasp_sampler._create_target_se3(
            pos=hole_world_pos, 
            approach_dir=straight_down_dir, 
            roll=0.0
        )

        centered_pos = place_world_SE3.translation - self.env.robot_base_pos
        hole_relative_pos = self.env.robot_base_mat.T @ centered_pos
        hole_relative_mat = self.env.robot_base_mat.T @ place_world_SE3.rotation
        place_base_SE3 = pin.SE3(hole_relative_mat, hole_relative_pos)

        place_approach_pos = hole_relative_pos + np.array([0, 0, 0.15])
        place_approach_SE3 = pin.SE3(hole_relative_mat, place_approach_pos)

        q_place_approach = self.ik.solve(q_init=q_pick_lift[:7], target_SE3=place_approach_SE3)
        q_place_drop = self.ik.solve(q_init=q_place_approach[:7], target_SE3=place_base_SE3)

        if q_place_approach is None or q_place_drop is None:
            print("IK failed for place approach or drop target configurations.")
            return None

        return q_place_approach, q_place_drop