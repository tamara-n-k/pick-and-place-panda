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
    WELDING = auto()
    LIFT_PICK = auto()
    
    APPROACH_PLACE = auto()
    DESCEND_PLACE = auto()
    UNWELDING = auto()
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
        self.weld_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_EQUALITY, "plant_attachment")


    def run(self):
        self.state = TaskState.OPEN_GRIPPER
        
        while self.state not in [TaskState.COMPLETED, TaskState.FAILED]:
            self.update_fsm()
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()

        return self.state == TaskState.COMPLETED


    def update_fsm(self):
        state_handlers = {
            TaskState.OPEN_GRIPPER:   self._handle_open_gripper,
            TaskState.APPROACH_PICK:  self._handle_approach_pick,
            TaskState.DESCEND_PICK:   self._handle_descend_pick,
            TaskState.GRASPING:       self._handle_grasping,
            TaskState.WELDING:        self._handle_welding,
            TaskState.LIFT_PICK:      self._handle_lift_pick,
            TaskState.APPROACH_PLACE: self._handle_approach_place,
            TaskState.DESCEND_PLACE:  self._handle_descend_place,
            TaskState.UNWELDING:      self._handle_unwelding,
            TaskState.RELEASE:        self._handle_release,
            TaskState.RETRACT_PLACE:  self._handle_retract_place,
        }

        handler = state_handlers.get(self.state)
        if handler:
            handler()


    def _handle_open_gripper(self):
        print("FSM: Opening gripper...")
        self.env.robot.set_gripper_ctrl(width=0.04)
        # Give the physical engine 50 frames to open up
        for _ in range(50):
            self.env.step()
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
        
        self.task.planner.set_collision_mask(self.plant_name, ignore=False)
        self.task.planner.set_collision_mask("tabletop", ignore=False)
        
        if success:
            self.state = TaskState.GRASPING
        else:
            self.state = TaskState.FAILED


    def _handle_grasping(self):
        print("FSM: Closing jaws...")
        self.env.robot.set_gripper_ctrl(width=0.0)
        
        # Lock arm joints while jaws contract
        for _ in range(200):
            current_gripper_q = self.env.get_state()[7:]
            anchor_state = np.concatenate((self.q_grasp[:7], current_gripper_q))
            self.env.set_state(anchor_state)
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()
        
        if self.check_grasp():
            print("FSM: Grasp contact validated visually and physically.")
            self.state = TaskState.WELDING
        else:
            print("CRITICAL: Jaws closed but contact validation failed.")
            self.state = TaskState.FAILED

    
    def _handle_welding(self):
        print("FSM: Securing spatial constraint welding lock...")
        self.env.robot.data.eq_active[self.weld_id] = 1
        self.env.robot.model.eq_data[self.weld_id, :7] = 0
        mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
        self.state = TaskState.LIFT_PICK


    def _handle_lift_pick(self):
        print("FSM: Elevating structural target payload...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=True)
        if self.task.execute_move(self.q_pick_lift, self.plant_name, linear=False):
            self.state = TaskState.APPROACH_PLACE
        else:
            self.state = TaskState.FAILED
    

    # placing sequence starts here
    def _handle_approach_place(self):
        print(f"FSM: Transit to target place approach above {self.tray_name}...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=True)
        if self.task.execute_move(self.q_place_approach, self.plant_name, linear=False):
            self.state = TaskState.DESCEND_PLACE
        else:
            self.state = TaskState.FAILED


    def _handle_descend_place(self):
        print(f"FSM: Linear insert descent into the tray hole...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=True)
        if self.task.execute_move(self.q_place_drop, self.plant_name, linear=True):
            self.state = TaskState.UNWELDING
        else:
            self.state = TaskState.FAILED


    def _handle_unwelding(self):
        print("FSM: Releasing MuJoCo spatial constraint weld lock...")
        self.env.robot.data.eq_active[self.weld_id] = 0
        mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
        self.state = TaskState.RELEASE


    def _handle_release(self):
        print("FSM: Opening jaws to release payload...")
        self.env.robot.set_gripper_ctrl(width=0.04)
        
        # Hold the position fixed while fingers expand
        for _ in range(150):
            current_gripper_q = self.env.get_state()[7:]
            anchor_state = np.concatenate((self.q_place_drop[:7], current_gripper_q))
            self.env.set_state(anchor_state)
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()
        
        self.state = TaskState.RETRACT_PLACE


    def _handle_retract_place(self):
        print("FSM: Retracting arm straight up back to safety...")
        self.task.planner.set_collision_mask(self.plant_name, ignore=False)
        if self.task.execute_move(self.q_place_approach, self.plant_name, linear=True):
            self.state = TaskState.COMPLETED
        else:
            self.state = TaskState.FAILED


    def check_grasp(self):
        f1_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        f2_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        box_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_BODY, self.plant_name)

        touch_f1, touch_f2 = False, False
        data = self.env.robot.data
        model = self.env.robot.model

        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = model.geom_bodyid[contact.geom1]
            b2 = model.geom_bodyid[contact.geom2]

            if (b1 == f1_id and b2 == box_id) or (b1 == box_id and b2 == f1_id):
                touch_f1 = True
            if (b1 == f2_id and b2 == box_id) or (b1 == box_id and b2 == f2_id):
                touch_f2 = True

        print(f"Grasp check: Left finger contact: {touch_f1}, Right finger contact: {touch_f2}")
        return touch_f1 and touch_f2