import time
from enum import Enum, auto
import numpy as np
import mujoco
import pinocchio as pin

class TaskState(Enum):
    IDLE = auto()
    OPEN_GRIPPER = auto()
    APPROACH_PICK = auto()
    DESCEND_PICK = auto()
    GRASPING = auto()
    GRASP_SETTLE = auto()
    LIFT_PICK = auto()
    
    APPROACH_PLACE = auto()
    DESCEND_PLACE = auto()
    RELEASE = auto()
    RETRACT_PLACE = auto()
    
    COMPLETED = auto()
    FAILED = auto()

class PickAndPlaceStateMachine:
    def __init__(self, task_runner, plant_name, tray_name, 
                 q_pick_approach, q_grasp, q_pick_lift,
                 q_place_approach, q_place_drop):
        
        self.task = task_runner
        self.env = task_runner.env
        self.plant_name = plant_name
        self.tray_name = tray_name
        
        # Pick joint configurations
        self.q_pick_approach = q_pick_approach
        self.q_grasp = q_grasp
        self.q_pick_lift = q_pick_lift
        
        # Place joint configurations
        self.q_place_approach = q_place_approach
        self.q_place_drop = q_place_drop
        
        self.state = TaskState.IDLE

    def run(self):
        self.state = TaskState.OPEN_GRIPPER
        
        while self.state not in [TaskState.COMPLETED, TaskState.FAILED]:
            self.update_fsm()
        return self.state == TaskState.COMPLETED


    def update_fsm(self):
        state_handlers = {
            TaskState.OPEN_GRIPPER:   self._handle_open_gripper,
            TaskState.APPROACH_PICK:  self._handle_approach_pick,
            TaskState.DESCEND_PICK:   self._handle_descend_pick,
            TaskState.GRASPING:       self._handle_grasping,
            TaskState.LIFT_PICK:      self._handle_lift_pick,
            TaskState.APPROACH_PLACE: self._handle_approach_place,
            TaskState.DESCEND_PLACE:  self._handle_descend_place,
            TaskState.RELEASE:        self._handle_release,
            TaskState.RETRACT_PLACE:  self._handle_retract_place,
        }

        handler = state_handlers.get(self.state)
        if handler:
            handler()


    def _handle_open_gripper(self):
        print("FSM: Opening gripper...")
        current_q = self.env.get_state()[:7]
        for _ in range(100): 
            self.env.robot.set_arm_ctrl(current_q)
            self.env.robot.set_gripper_ctrl(width=0.04)# set gripper control
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()
        self.state = TaskState.APPROACH_PICK
    

    def _handle_approach_pick(self):
        print(f"FSM: Approaching {self.plant_name}...")
        if self.task.execute_move(self.q_pick_approach, self.plant_name, linear=False):
            self.state = TaskState.DESCEND_PICK
        else:
            self.state = TaskState.FAILED


    def _handle_descend_pick(self):
        print(f"FSM: Descending onto {self.plant_name}...")            
        success = self.task.execute_move(self.q_grasp, self.plant_name, linear=True)       
        if success:
            self.state = TaskState.GRASPING
        else:
            self.state = TaskState.FAILED


    def _handle_grasping(self):
        print("FSM: Closing jaws...")
        for _ in range(500):
            self.env.robot.set_arm_ctrl(self.q_grasp[:7])
            self.env.robot.set_gripper_ctrl(width=0.0)
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()
        
        if self.check_grasp():
            print("FSM: Grasp verified successfully!")
            self.state = TaskState.LIFT_PICK
        else:
            print("FSM: CRITICAL - Grasp failed validation check.")
            self.state = TaskState.FAILED

    
    def _handle_lift_pick(self):
        print("FSM: Elevating structural target payload...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=True)
        if self.task.execute_move(self.q_pick_lift, self.plant_name, linear=False, gripper_width=0.0):
            self.state = TaskState.APPROACH_PLACE
        else:
            self.state = TaskState.FAILED
    

    # placing sequence starts here
    def _handle_approach_place(self):
        print(f"FSM: Transit to target place approach above {self.tray_name}...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=True)
        if self.task.execute_move(self.q_place_approach, self.plant_name, linear=False, gripper_width=0.0):
            self.state = TaskState.DESCEND_PLACE
        else:
            self.state = TaskState.FAILED


    def _handle_descend_place(self):
        print(f"FSM: Linear insert descent into the tray hole...")
        if self.task.execute_move(self.q_place_drop, self.plant_name, linear=True, gripper_width=0.0):
            self.state = TaskState.RELEASE
        else:
            self.state = TaskState.FAILED


    def _handle_release(self):
        print("FSM: Opening jaws to release payload...")        
        for _ in range(500):
            self.env.robot.set_arm_ctrl(self.q_place_drop[:7])
            self.env.robot.set_gripper_ctrl(0.04)
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()
        self.state = TaskState.RETRACT_PLACE


    def _handle_retract_place(self):
        print("FSM: Retracting arm straight up back to safety...")
        if self.task.execute_move(self.q_place_approach, self.plant_name, linear=True, gripper_width=0.04):
            self.state = TaskState.COMPLETED
        else:
            self.state = TaskState.FAILED


    def check_grasp(self):
        data = self.env.robot.data
        model = self.env.robot.model

        touch_f1, touch_f2 = False, False

        # Loop through every active contact point
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # get body ids
            b1_id = model.geom_bodyid[contact.geom1]
            b2_id = model.geom_bodyid[contact.geom2]
            
            # convert to body names
            b1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1_id)
            b2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2_id)

            if b1_name and b2_name:
                if ("finger" in b1_name and self.plant_name in b2_name) or ("finger" in b2_name and self.plant_name in b1_name):
                    if "left" in b1_name or "left" in b2_name:
                        touch_f1 = True
                    if "right" in b1_name or "right" in b2_name:
                        touch_f2 = True

        print(f"Grasp check: Left finger contact: {touch_f1}, Right finger contact: {touch_f2}")
        return touch_f1 and touch_f2
