import numpy as np
import time
import mujoco.viewer
from robot.robot import Robot

class Viewer:
    """
    executes the mujoco simulator
    """
    @staticmethod
    def run_trajectory(env, trajectory, record=True, draw=True):
        executed_positions = []

        with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as viewer:
            for i in range(trajectory.getStateCount()):
                state = trajectory.getState(i)
                env.apply_action(state)

                if record:
                    ee_pos = env.robot.data.site("gripper").xpos.copy()
                    executed_positions.append(ee_pos)
                # draw trajectory
                if draw:
                    viewer.user_scn.ngeom = 0 #clear previous geometrics
                    for p in executed_positions:
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.008, 0, 0],  # radius = 8 mm
                            pos=p,
                            mat=np.eye(3).flatten(),
                            rgba=[1, 0, 0, 0.8],  # red, semi-transparent
                        )
                        viewer.user_scn.ngeom += 1
                #draw goal
                if hasattr(env, "goal_cartesian") and env.goal_cartesian is not None:
                    goal_pos = env.goal_cartesian.translation
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0], #radius = 20 mm
                        pos=goal_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 1],
                    )
                    viewer.user_scn.ngeom += 1
                viewer.sync()
                time.sleep(0.01)

            # Keep window open
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
        if record:
            return np.array(executed_positions)
