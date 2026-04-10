import numpy as np
import time
import mujoco.viewer
from robot.mj_robot import MJRobot

class Viewer:
    """
    executes the mujoco simulator
    """
    @staticmethod
    def run_trajectory(env, trajectory, speed = 1.0, record=True, draw=True):
        executed_positions = []

        viewer_handle = env.mj_viewer

        if not viewer_handle or not viewer_handle.is_running():
            print("Viewer is not running. Cannot execute trajectory.")
            return
        
        print("Starting viewer...")
        for state in trajectory:
            env.set_state(state)
            if record:
                try:
                    ee_pos = env.robot.data.site("gripper").xpos.copy()
                    executed_positions.append(ee_pos)
                except Exception as e:
                    print(f"Error occurred while recording position: {e}")

            if draw:
                viewer_handle.user_scn.ngeom = 0 #clear previous geometrics

            # # --------visualization--------
            # #draw goal
            # if hasattr(env, "goal_cartesian") and env.goal_cartesian is not None:
            #     goal_pos = env.goal_cartesian.translation
            #     mujoco.mjv_initGeom(
            #         viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom],
            #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #         size=[0.05, 0, 0], #radius = 50 mm
            #         pos=goal_pos,
            #         mat=np.eye(3).flatten(),
            #         rgba=[0, 1, 0, 1],
            #     )
            #     viewer_handle.user_scn.ngeom += 1

            #     # executed path
            #     for p in executed_positions:
            #         mujoco.mjv_initGeom(
            #             viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom],
            #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #             size=[0.008, 0, 0],  # radius = 8 mm
            #             pos=p,
            #             mat=np.eye(3).flatten(),
            #             rgba=[1, 0, 0, 0.8],  # red, semi-transparent
            #         )
            #         viewer_handle.user_scn.ngeom += 1
            # # ----------------------------------


            viewer_handle.sync()
            time.sleep(0.01/speed)
            
        last_ee_pos = executed_positions[-1]
        # print("Final executed EE position:", last_ee_pos)
        # print("Goal position:", env.goal_cartesian.translation)

       
        if record:
            return np.array(executed_positions)
