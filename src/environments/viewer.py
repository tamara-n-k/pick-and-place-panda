import numpy as np
import time
import mujoco.viewer
from robot.mj_robot import MJRobot
import pinocchio as pin

class Viewer:
    """
    executes the mujoco simulator
    """
    @staticmethod
    def setup_initial_view(env):
        if env.mj_viewer is not None:
            with env.mj_viewer.lock():
                env.mj_viewer.cam.type = 0  # free camera (default)
                env.mj_viewer.cam.distance = 3.5  
                env.mj_viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
                env.mj_viewer.cam.elevation = -30
                env.mj_viewer.cam.azimuth = 135
            env.mj_viewer.sync()

    @staticmethod
    def set_collision_visualization(env, enabled=True):
        viewer = env.mj_viewer
        if viewer is None:
            return

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(enabled)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = int(enabled)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = int(enabled)
            viewer.opt.geomgroup[3] = int(enabled)

        viewer.sync()
