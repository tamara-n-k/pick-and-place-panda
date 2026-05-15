import time
from enum import Enum, auto
import numpy as np
import mujoco
import pinocchio as pin

class PickState(Enum):
    IDLE = auto()
    OPEN_GRIPPER = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASPING = auto()
    WELDING = auto()
    LIFTING = auto()
    COMPLETED = auto()
    FAILED = auto()

class PickStateMachine:
    def __init__(self, task_runner, plant_name, q_grasp, q_approach, q_lift):
        self.task = task_runner
        self.env = task_runner.env
        self.plant_name = plant_name
        
        self.q_grasp = q_grasp
        self.q_approach = q_approach
        self.q_lift = q_lift
        
        self.state = PickState.IDLE
        self.weld_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_EQUALITY, "plant_attachment")

    def run(self):
        self.state = PickState.OPEN_GRIPPER
        
        while self.state not in [PickState.COMPLETED, PickState.FAILED]:
            self.update_fsm()
            self.env.step()
            if self.env.mj_viewer.is_running():
                self.env.mj_viewer.sync()

        return self.state == PickState.COMPLETED

    def update_fsm(self):
        # open gripper
        if self.state == PickState.OPEN_GRIPPER:
            print("FSM: Opening gripper...")
            self.env.robot.set_gripper_ctrl(width=0.04)
            # Give the physical engine 50 frames to open up
            for _ in range(50):
                self.env.step()
            self.state = PickState.APPROACH

        # approach target
        elif self.state == PickState.APPROACH:
            print(f"FSM: Approaching {self.plant_name}...")
            if self.task.execute_move(self.q_approach, self.plant_name, linear=False):
                self.state = PickState.DESCEND
            else:
                self.state = PickState.FAILED

        # descend linearly onto target
        elif self.state == PickState.DESCEND:
            print(f"FSM: Descending onto {self.plant_name}...")
            # Ignore collisions specifically during descent so we clear bounds safely
            # self.task.planner.set_collision_mask(self.plant_name, ignore=True)
            # self.task.planner.set_collision_mask("tabletop", ignore=True)
            
            success = self.task.execute_move(self.q_grasp, self.plant_name, linear=True)
            
            self.task.planner.set_collision_mask(self.plant_name, ignore=False)
            self.task.planner.set_collision_mask("tabletop", ignore=False)
            
            if success:
                self.state = PickState.GRASPING
            else:
                self.state = PickState.FAILED

        # grasp target
        elif self.state == PickState.GRASPING:
            print("FSM: Closing jaws...")
            self.env.robot.set_gripper_ctrl(width=0.0)
            
            # Wait for gripper to stall or reach target position
            stalled = False
            for _ in range(200):
                self.env.step()
                if self.is_gripper_stalled():
                    stalled = True
                    break
            
            if self.check_grasp():
                print("FSM: Grasp contact validated visually and physically.")
                self.state = PickState.WELDING
            else:
                print("CRITICAL: Jaws closed but contact validation returned False.")
                self.state = PickState.FAILED

        # weld
        elif self.state == PickState.WELDING:
            print("FSM: Securing spatial constraint welding lock...")
            self.env.robot.data.eq_active[self.weld_id] = 1
            self.env.robot.model.eq_data[self.weld_id, :7] = 0
            mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
            self.state = PickState.LIFTING

        # lift
        elif self.state == PickState.LIFTING:
            print("FSM: Elevating structural target payload...")
            # Mask object during escape lift so the planner doesn't trip over its new payload
            self.task.planner.set_collision_mask(self.plant_name, ignore=True)
            
            if self.task.execute_move(self.q_lift, self.plant_name, linear=False):
                self.state = PickState.COMPLETED
            else:
                self.state = PickState.FAILED

    def is_gripper_stalled(self, minimum_frames=40, threshold=1e-4):
        if self.timer < minimum_frames:
            return False # Too early to check stall!
            
        v1 = self.env.robot.data.joint('finger_joint1').qvel[0]
        v2 = self.env.robot.data.joint('finger_joint2').qvel[0]
        return abs(v1) < threshold and abs(v2) < threshold

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
                
        return touch_f1 and touch_f2