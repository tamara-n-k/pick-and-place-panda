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
    def run_trajectory(env, trajectory, grasp_SE3=None, speed = 1.0, show_col_obj=True, record=True, show_trajectory=True, show_planning_tcp=True, show_goal_pos=True):
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
            try:
                tcp_pose = env.pin.forward_kinematics(state[:7])
                if show_planning_tcp:
                    # Get the mocap ID for the visual target
                    mocap_id = env.robot.model.body('tcp_target').mocapid[0]
                    env.robot.data.mocap_pos[mocap_id] = tcp_pose.translation
                    env.robot.data.mocap_quat[mocap_id] = pin.SE3ToXYZQUAT(tcp_pose)[3:]
                if show_goal_pos:
                    if grasp_SE3 is not None:
                        goal_id = env.robot.model.body('grasp_goal').mocapid[0]
                        env.robot.data.mocap_pos[goal_id] = grasp_SE3.translation
                        env.robot.data.mocap_quat[goal_id] = pin.SE3ToXYZQUAT(grasp_SE3)[3:]
            except Exception as e:
                pass

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
