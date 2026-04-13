import numpy as np
import time
import mujoco.viewer
from robot.mj_robot import MJRobot

class Viewer:
    """
    executes the mujoco simulator
    """
    @staticmethod
    def run_trajectory(env, trajectory, speed = 1.0, show_col_obj=False, record=True, show_trajectory=True):
        viewer_handle = env.mj_viewer
        if not viewer_handle or not viewer_handle.is_running():
            print("Viewer is not running. Cannot execute trajectory.")
            return
        if show_col_obj:
            # Show contact points (red arrows) and collision geometries
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1 
        else:
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0


        executed_positions = []

        print("Starting viewer...")
        for state in trajectory:
            env.set_state(state)
            if record:
                try:
                    # record end-effector position for visualization
                    ee_pos = env.robot.data.site("gripper").xpos.copy()
                    executed_positions.append(ee_pos)
                except Exception as e:
                    print(f"Error occurred while recording position: {e}")

            if show_trajectory:
                viewer_handle.user_scn.ngeom = 0 #clear previous geometrics

                for p in executed_positions:
                        mujoco.mjv_initGeom(
                            viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.005, 0, 0], # 5mm dots
                            pos=p,
                            mat=np.eye(3).flatten(),
                            rgba=[1, 0, 0, 0.5], # Red semi-transparent
                        )
                        viewer_handle.user_scn.ngeom += 1

            viewer_handle.sync()
            time.sleep(0.01 / speed)
            
        last_ee_pos = executed_positions[-1]
        print("Final executed EE position:", last_ee_pos)

        return np.array(executed_positions)
