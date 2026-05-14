import mujoco
import numpy as np
import pinocchio as pin
import time

from environments.viewer import Viewer
from planners.grasp_planner import GraspSampler

class PickAndPlace:
    def __init__(self, env, planner, ik_solver):
        self.env = env
        self.planner = planner
        self.ik = ik_solver
        self.grasp_sampler = GraspSampler(self.env, self.ik, self.planner)
        self.approach_margin = 0.1


    def execute_move(self, target_conf, plant_name, grasp_SE3=None, linear=False):
        current_q_full = self.env.get_state()
        q_start_arm = current_q_full[:7]
        q_gripper = current_q_full[7:]
        if not self.planner.collision_free(current_q_full):
            print("CRITICAL: RRT cannot start because the CURRENT robot position is in collision!")
        if isinstance(target_conf, np.ndarray) and target_conf.shape == (3,):
            target_SE3 = pin.SE3.Identity()
            target_SE3.translation = target_conf.astype(np.float64).flatten()
            target_SE3.rotation = pin.utils.rotate('x', np.pi) 
            q_goal_arm = self.ik.solve(q_start_arm, target_SE3)
        else:
            q_goal_arm = target_conf[:7]
        
        if q_goal_arm is None:
            print(f"IK failed for position {target_conf}")
            return False
        
        path_list_7dof = []

        # turn mask on before planning
        self.planner.set_collision_mask(plant_name, ignore=True)

        if linear:
            print(f"Executing linear move to {q_goal_arm}")
            path_arm = self.planner.find_linear_path(q_start_arm, q_goal_arm, plant_name, path_list_7dof)
            if path_arm is None:
                print(f"Linear Planning failed for target position: {target_conf}")
                return False
        else:
            print(f"Executing RRT move to {target_conf}")
            path_arm = self.planner.find_path_rrt(q_start_arm, q_goal_arm, plant_name)
            if path_arm is None:
                print(f"RRT Planning failed for target position: {target_conf}")
                return False
            pdef = self.planner.pdef

            if pdef.hasExactSolution():
                ompl_states = path_arm.getStates()
                path_list_7dof = []
                for i in range(len(ompl_states)):
                    state_7dof = np.array([ompl_states[i][j] for j in range(7)]) # extract only the arm joints
                    path_list_7dof.append(state_7dof)
                    
        # turn mask off immendiately after planning so we can execute the trajectory with correct collision checking
        self.planner.set_collision_mask(plant_name, ignore=False)

        if len(path_list_7dof) > 0:
            full_path = [np.concatenate((q_arm, q_gripper)) for q_arm in path_list_7dof]
            self.env.set_state(full_path[-1])
            for q in full_path:
                self.env.set_state(q)
                self.env.step()
                if self.env.mj_viewer.is_running():
                    time.sleep(0.01) # slow down the execution for visualization
                    self.env.mj_viewer.sync()

            return True
        
        print(f"No valid path found for target position: {target_conf}")
        return False



    def pick_plant(self, plant_name):
        current_q = self.env.get_state()
        print(f"--- Sampling valid grasp for {plant_name} ---")
        q_grasp, grasp_SE3 = self.grasp_sampler.find_valid_grasp(plant_pos=self.env.get_body_pos(plant_name), plant_name=plant_name)
        if q_grasp is None:
            print(f"No valid grasp found for {plant_name}")
            return False
        print(f"Found valid grasp for {plant_name} at position {grasp_SE3.translation}")
        approach_pos = grasp_SE3.translation + np.array([0, 0, 0.15]) 
        approach_SE3 = pin.SE3(grasp_SE3.rotation, approach_pos)
        q_approach = self.ik.solve(current_q[:7], approach_SE3)
        # open gripper
        if not self.env.robot.set_gripper_ctrl(width=0.04):
            print("Failed to open gripper.")
            return
        for _ in range(50):
            self.env.step()
        
        # approach
        print(f"--- Picking {plant_name} ---")
        
        if not self.execute_move(q_approach, plant_name, grasp_SE3):
            print("Failed to approach plant.")
            return
        print(f"--- Reached approach pose {approach_SE3.translation} for {plant_name} ---")
        
        # descend
        print(f"--- Descending to grasp {plant_name} ---")
        if not self.execute_move(q_grasp, plant_name, grasp_SE3, True):
            print("Failed to descend to grasp plant.")
            return

        # close Gripper
        print(f"--- Grasping {plant_name} ---")
        if not self.env.robot.set_gripper_ctrl(width=0.0):
            print("Failed to close gripper.")
            return
        for _ in range(100):
            self.env.step()

        weld_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_EQUALITY, "plant_attachment")
        # activate the weld constraint to attach the plant to the gripper
        self.env.robot.data.eq_active[weld_id] = 1
        print(f"Successfully grasped {plant_name}!")
        # set the weld distance to 0 to ensure the plant follows the gripper perfectly
        self.env.robot.model.eq_data[weld_id, :7] = 0
        # lift
        q_lift = self.ik.solve(q_init=q_grasp, target_SE3=approach_SE3)
        print(f"--- Lifting {plant_name} ---")
        if not self.execute_move(q_lift, plant_name, grasp_SE3):
            print("Failed to lift plant.")
            return
