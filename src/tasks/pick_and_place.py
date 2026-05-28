import mujoco
import numpy as np
import pinocchio as pin
import time
from planners.grasp_planner import GraspSampler
from fsm.pick_and_place_fsm import PickAndPlaceStateMachine

class PickAndPlace:
    def __init__(self, env, planner, ik_solver):
        self.env = env
        self.planner = planner
        self.ik = ik_solver
        self.grasp_sampler = GraspSampler(self.env, self.ik, self.planner)
        self.approach_margin = 0.1
        self.place_offset_index = 0

    def execute_move(self, q_goal_arm, plant_name, linear=False, gripper_width=None):
        q_start_arm = self.env.get_arm_state()
        
        if not self.planner.collision_free(q_start_arm):
            print("CRITICAL: Planner reports initialization state is in structural collision!")

        path_list_7dof = []
        self.planner.set_collision_mask(plant_name, ignore=True)

        if linear:
            print(f"Executing linear path to: {q_goal_arm}")
            path_list_7dof = self.planner.find_linear_path(q_start_arm, q_goal_arm, plant_name)
            if path_list_7dof is None:
                self.planner.set_collision_mask(plant_name, ignore=False)
                return False
        else:
            print(f"Executing RRT path to: {q_goal_arm}")
            path_arm = self.planner.find_path_rrt(q_start_arm, q_goal_arm, plant_name)
            if path_arm is None:
                self.planner.set_collision_mask(plant_name, ignore=False)
                return False
            
            pdef = self.planner.pdef
            if pdef.hasExactSolution() or path_arm is not None:
                try:
                    ompl_states = path_arm.getStates()
                    path_list_7dof = [np.array([ompl_states[i][j] for j in range(7)]) for i in range(len(ompl_states))]
                except AttributeError:
                    path_list_7dof = path_arm
                        
        self.planner.set_collision_mask(plant_name, ignore=False)

        if len(path_list_7dof) > 0:
            for q_arm in path_list_7dof:
                if not self.env.servo_to(q_arm, gripper_width=gripper_width):
                    print(f"Servo failed to reach waypoint. Error: {np.linalg.norm(self.env.get_arm_state() - q_arm):.4f}")
                    return False

            return True
        
        return False


    def execute_pick_and_place(self, plant_name, tray_name, hole_offsets):
        current_q = self.env.get_state()
        selected_hole_offset = self._select_place_offset(hole_offsets)
        
        pick_configs = self._compute_pick_configurations(plant_name, current_q)
        if pick_configs is None:
            return False
        q_grasp, q_pick_approach, q_pick_lift = pick_configs

        place_configs = self._compute_place_configurations(tray_name, selected_hole_offset, q_pick_lift)
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


    def _select_place_offset(self, hole_offsets):
        offsets = np.asarray(hole_offsets, dtype=float)
        if offsets.shape == (3,):
            return offsets
        if offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) == 0:
            raise ValueError("Expected a single hole offset with shape (3,) or a list/array of offsets with shape (N, 3)")

        index = self.place_offset_index % len(offsets)
        selected_offset = offsets[index]
        self.place_offset_index += 1
        print(f"Selected place offset {index + 1}/{len(offsets)}: {selected_offset}")
        return selected_offset
    

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
        # get world position of place down tray
        tray_world_pos = self.env.robot.data.body(tray_name).xpos
        tray_world_mat = self.env.robot.data.body(tray_name).xmat.reshape(3, 3)
        # apply hole offset
        hole_world_pos = tray_world_pos + (tray_world_mat @ hole_offset_local)

        # use straight down approach
        straight_down_dir = np.array([0.0, 0.0, -1.0])
        place_world_SE3 = self.grasp_sampler._create_target_se3(
            pos=hole_world_pos, 
            approach_dir=straight_down_dir, 
            roll=0.0
        )
        # get offset from robot base
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
