import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial import KDTree
import time

model_path = "model/franka_emika_panda/pick_place_scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def is_collision(q):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return data.ncon > 0  # number of contacts

class Node:
    def __init__(self, q, parent=None):
        self.q = q
        self.parent = parent

def rrt(start, goal, joint_limits, is_collision, max_iter=1000, step_size=0.1):
    tree = [Node(start)]
    for _ in range(max_iter):
        rand_q = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
        
        # Find nearest node
        dists = [np.linalg.norm(n.q - rand_q) for n in tree]
        nearest = tree[np.argmin(dists)]

        direction = rand_q - nearest.q
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        new_q = nearest.q + step_size * direction / norm
        new_q = np.clip(new_q, [low for (low, _) in joint_limits], [high for (_, high) in joint_limits])

        if is_collision(new_q):
            continue

        new_node = Node(new_q, nearest)
        tree.append(new_node)

        if np.linalg.norm(new_q - goal) < step_size:
            print("Goal reached")
            goal_node = Node(goal, new_node)
            tree.append(goal_node)
            break

    # Backtrack
    path = []
    node = tree[-1]
    while node:
        path.append(node.q)
        node = node.parent
    return path[::-1]

# Launch interactive viewer
start = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5, 0, 0])   # example
goal = np.array([0.5, -0.2, 0.3, -1.0, 0.2, 1.2, -0.3, 0, 0])   # example

joint_limits = [
    (model.jnt_range[i][0], model.jnt_range[i][1])
    for i in range(model.njnt)
]
path = rrt(start, goal, joint_limits, is_collision)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for q in path:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        time.sleep(0.05)
        viewer.sync()