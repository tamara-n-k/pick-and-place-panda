import time
import mujoco.viewer
from robot.robot import Robot

class Viewer:
    @staticmethod
    def start_viewer(robot: Robot, path):
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            for i in range(path.getStateCount()):
                state = path.getState(i)
                for j in range(robot.n_dofs):
                    robot.data.qpos[j] = state[j]#passes the joint position of the planner to the model
                mujoco.mj_forward(robot.model, robot.data)
                viewer.sync()
                time.sleep(0.01)

            # Keep window open
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
