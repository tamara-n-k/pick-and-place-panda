import time
import mujoco.viewer
from robot.robot import Robot

class Viewer:
    """
    executes the mujoco simulator
    """
    @staticmethod
    def run_trajectory(env, trajectory):
        with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as viewer:
            for i in range(trajectory.getStateCount()):
                state = trajectory.getState(i)
                env.apply_action(state)
                viewer.sync()
                time.sleep(0.01)

            # Keep window open
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
