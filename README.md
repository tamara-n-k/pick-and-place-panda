# Franka Emika Panda Simulation
## Overview

This project implements a MuJoCo simulation of a Franka Emika Panda robot for object manipulation. The robot picks plant dummy objects from a tray and places them into predefined holes in another tray.

The system uses MuJoCo for simulation, Pinocchio for kinematics and self-collision checking, OMPL for motion planning, and a finite state machine to execute the full pick-and-place sequence.

## Setup
Ubuntu-Version 24.04
1. install mujoco via pip
2. install ompl with python wheels
pip install wheel
https://ompl.kavrakilab.org/installation.html
pip install ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl 
OR use environment.yml


## Required model files

- `model/franka_emika_panda/table_panda_scene.xml`
  - Main MuJoCo scene used by `src/main.py`.
  - Defines the Panda robot, table, plants, tray, joints, actuators, bodies, geoms, and collision objects.

- `model/franka_emika_panda/panda.urdf`
  - Robot description used by Pinocchio.
  - Required for inverse kinematics and self-collision checking.

- `model/franka_emika_panda/assets/`
  - Mesh folder used by both MuJoCo and Pinocchio.
  - Contains robot link meshes, gripper meshes, plant meshes, tray meshes, and other object geometry.
  - The XML and URDF files reference these meshes, so this folder must stay in the expected location.


## Pick-and-Place Pipeline

The project simulates a Franka Emika Panda robot performing a pick-and-place task in MuJoCo. The robot picks plant objects from the table and places them into predefined tray holes.

### Step-by-Step Execution

1. **Initialize the simulation**

   `main.py` loads the MuJoCo XML scene, the Panda URDF model, and the mesh assets. It creates the `Environment`, moves the robot to its home pose, and starts the passive MuJoCo viewer.

2. **Create planning and control modules**

   The main script creates:
   - `MotionPlanner` for collision-aware path planning
   - `IKSolver` for inverse kinematics
   - `PickAndPlace` for coordinating the full task

3. **Select an object and target hole**

   For each plant object, the task selects one tray hole offset. The offset defines where the object should be placed relative to the tray body.

4. **Compute a valid grasp**

   `GraspSampler` samples multiple candidate grasp poses around the plant. For each candidate, it:
   - creates an end-effector pose
   - solves inverse kinematics
   - checks collisions
   - scores the grasp based on clearance, alignment, joint limits, and approach angle

   The best valid grasp is selected.

5. **Compute pick configurations**

   After finding a grasp, the system computes:
   - a pick approach pose above the plant
   - the actual grasp pose
   - a lifted pose after grasping

6. **Compute place configurations**

   The tray hole position is converted into the robot base frame. The system then computes:
   - a place approach pose above the tray hole
   - a final drop pose inside the tray

7. **Run the finite state machine**

   The `PickAndPlaceStateMachine` executes the task in ordered states:
   - open gripper
   - approach pick position
   - descend to grasp
   - close gripper
   - verify grasp using MuJoCo contact data
   - lift object
   - move above tray
   - descend into place position
   - open gripper to release
   - retract upward
   - return home

8. **Plan and execute motion**

   For long movements, the `MotionPlanner` uses OMPL RRTConnect to find a collision-free joint-space path. For vertical pick/place motions, it uses linear interpolation between joint configurations.

9. **Servo through the trajectory**

   The environment sends each planned joint waypoint to the MuJoCo position controllers. During each simulation step, gravity compensation is applied and the viewer is synchronized.

10. **Finish or fail the task**

   If all states succeed, the FSM returns `COMPLETED`. If IK, planning, collision checking, servoing, or grasp validation fails, the FSM returns `FAILED`.


## Some Theory

**Joint ID vs. Frame ID**

A joint describes a degree of freedom in the kinematic chain (rotation, translation, etc.).

A frame is a coordinate system attached to a joint, link, or custom location in your robot model.

**Body vs. Site**

A **body** is a rigid object in the model. Each body has:
- a position
- an orientation
- physical properties like mass, intertia

They define the structure of the robot: links, base, end-effector etc.

A **site** is just a fixed coordinate frame (position + orientation) attached to a body. It has no mass, no inertia and doesn't affect physics. We use it to track a specific location on the robot.

Example:
  
    <body name="hand">
    <site name="ee_site" pos="0 0 0.05" size="0.01 0.01 0.01" rgba="1 0 0 1"/>
    </body>

**Scaling:**

In mujoco the unit scale is 1.0 m. So when loading a CAD model, it has to be imported with the scale parameter: ``` scale="0.001 0.001 0.001"```.For for primitive shapes like ```<geom type="box"> ``` mujoco will be using half-length parameters, so every scale you give has to be divided by 2 to get the actual size. 
E. g. you want to define a box with 1.0 m lenght and 1.0 m width, you define it like this:
```<geom type="box" size="0.5 0.5 0.05"/>```

**Contact and Collision Parameters in MuJoCo:**

MuJoCo uses contact parameters on `geom` elements to decide which objects can collide and how those contacts behave during simulation. These parameters are important for this project because the robot must avoid the table and tray, but still be able to touch and grasp the plants.

**Bitmasks: `contype` and `conaffinity`**

`contype` and `conaffinity` are bitmasks used to filter contacts between geoms.

- `contype`: the collision category that this geom belongs to
- `conaffinity`: the collision categories that this geom is allowed to collide with

A contact is generated if this condition is true:

```text
(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity) != 0
```

The `&` operator compares the bits of the two numbers. If at least one matching bit is found, MuJoCo allows the two geoms to collide.

Examples:

```xml
<geom contype="1" conaffinity="1"/>
```

This geom belongs to collision group `1` and collides with other geoms that also accept group `1`.

```xml
<geom contype="0" conaffinity="0"/>
```

This geom does not generate contacts. It is usually used for visual-only meshes, debug markers, or objects that should be visible but not physically collide.

In this project, visual meshes often use `contype="0"` and `conaffinity="0"`, while collision geoms use active values such as `contype="1"` and `conaffinity="1"`.

**`condim`**

`condim` controls the number of contact constraint dimensions MuJoCo uses for a contact.

- `condim="1"`: normal force only
- `condim="3"`: normal force plus sliding friction
- higher values can include torsional and rolling friction

For grasping, `condim="3"` is useful because the fingers need friction against the object, not just a normal pushing force.

Example from the scene:

```xml
<geom condim="3" friction="1.5 0.005 0.0001"/>
```

This allows the contact to use sliding friction, which helps the gripper hold the plant instead of letting it slip easily.

**`friction`**

The `friction` parameter defines friction behavior for a geom:

```xml
friction="sliding torsional rolling"
```

For example:

```xml
friction="1.5 0.005 0.0001"
```

- `1.5`: sliding friction
- `0.005`: torsional friction
- `0.0001`: rolling friction

Higher sliding friction makes it easier for the gripper to hold an object.

**`solref`**

`solref` controls how stiff and damped the contact constraint is.

```xml
solref="timeconst dampratio"
```

Example:

```xml
solref="0.01 1"
```

- the first value controls how quickly MuJoCo corrects contact errors
- the second value controls damping

Smaller time constants usually make contacts stiffer and more rigid. Larger values make contacts softer. A damping ratio around `1` is close to critically damped behavior, which helps avoid excessive oscillation.

**`solimp`**

`solimp` controls contact impedance. In simple terms, it defines how strongly MuJoCo enforces a constraint as penetration or constraint error changes.

Example:

```xml
solimp="0.95 0.99 0.001"
```

The first two values define the minimum and maximum impedance. Values close to `1` make the contact behave more rigidly. The third value controls the transition width between soft and hard behavior.

In this project, values such as:

```xml
solimp="0.95 0.99 0.001"
solref="0.01 1"
```

create relatively firm contacts, which is useful for stable grasping and placement.

**How this project uses contact settings**

- Visual robot meshes can be disabled for collision with `contype="0"` and `conaffinity="0"`.
- Simplified collision geoms are used for physical contacts.
- Plant and tray geoms are collision-enabled so the planner can detect invalid robot-object contacts.
- During grasp planning and execution, the code can temporarily change collision masks with `set_collision_mask()` so the gripper is allowed to approach or hold the target plant without the planner rejecting every close grasp.

**Collision Checking:**

The collision_free method of the Motion Planner uses mujoco data to check where the plants, the table and the trays are. It calculates physical contacts between the robots meshes and the objects defined in the xml. The collision_free method is used by the Grasp Sampler and the Motion Planner.
The check_collisions function in the Pin Robot only checks for self collisions using pinocchio forward kinematics.
