"""Helper functions for the control lab.

You do not need to edit this file. These helpers only set things up and draw plots:
the control loops, the controllers and the dynamics calls are written in the notebook.
"""
import os
import sys
import urllib.request

import numpy as np
import pybullet as p
import pybullet_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IN_COLAB = "google.colab" in sys.modules
REPO_RAW = "https://raw.githubusercontent.com/LxViRaL-teaching/medical-robotics-2026/main/week-3/"

DT = 1.0 / 240.0   # simulation time step [s], also our control period
G = 9.81           # gravity [m/s^2], the same value in the simulator and in your calculations

_MODELS = {}       # robot id -> its hidden copy used for inverse dynamics (see change_mass)


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def fetch(filename):
    """Download a lab file from the course repository if it is not in this folder (Colab)."""
    if not os.path.exists(filename):
        urllib.request.urlretrieve(REPO_RAW + filename, filename)
        print("downloaded", filename)


def connect(camera_target=(0.3, -0.25, 1.0), camera_distance=3.0, yaw=0, pitch=-15):
    """Start PyBullet with gravity ON and time step DT (3D window locally, no window on Colab).

    If a simulation is already running from an earlier cell, it is closed first.
    """
    _close_models()
    if p.isConnected():
        p.disconnect()
    client = p.connect(p.DIRECT if IN_COLAB else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -G)
    p.setTimeStep(DT)
    if not IN_COLAB:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=yaw,
                                     cameraPitch=pitch, cameraTargetPosition=list(camera_target))
    return client


def load_robot(urdf_file, base_position=(0, 0, 0)):
    """Load a URDF with a fixed base, ready for torque control. Returns the body id.

    Three things are done for you:
      * the masses and inertias are taken from the URDF file as written (by default PyBullet
        recomputes the inertias from the collision shapes);
      * PyBullet's default link damping is set to zero, so the only losses are the joint damping
        and friction written in the URDF;
      * the default velocity motor on every joint is switched off (force 0). Otherwise it acts as
        a brake and your torque commands are not the only torques on the joint.
    """
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(urdf_file, list(base_position), useFixedBase=True,
                       flags=p.URDF_USE_INERTIA_FROM_FILE)
    for j in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
    ids, _ = moving_joints(robot)
    p.setJointMotorControlArray(robot, ids, p.VELOCITY_CONTROL, forces=[0.0] * len(ids))
    _MODELS[robot] = {"urdf": urdf_file, "base": list(base_position), "masses": {},
                      "client": None, "body": None}
    return robot


# ----------------------------------------------------------------------------
# Joints and links (by NAME, never by index)
# ----------------------------------------------------------------------------
def moving_joints(robot):
    """Indices and names of the non-fixed joints, in URDF order."""
    ids, names = [], []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] != p.JOINT_FIXED:
            ids.append(j)
            names.append(info[1].decode())
    return ids, names


def joint_index(robot, joint_name):
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[1].decode() == joint_name:
            return j
    raise KeyError(f"no joint called '{joint_name}'")


def link_index(robot, link_name):
    """In PyBullet a link has the same index as the joint that connects it to its parent."""
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[12].decode() == link_name:
            return j
    raise KeyError(f"no link called '{link_name}'")


def reset_joints(robot, joint_ids, q):
    """Teleport the joints to positions q, with zero velocity."""
    for j, qi in zip(joint_ids, np.atleast_1d(q)):
        p.resetJointState(robot, j, float(qi), 0.0)


def get_mass(robot, link_name):
    """Mass of a link [kg], as the simulator currently has it."""
    return p.getDynamicsInfo(robot, link_index(robot, link_name))[0]


def get_length(robot, link_name):
    """Distance [m] from the frame of the link's parent to the frame of `link_name`.

    For the pendulum, get_length(pendulum, "ball_link") is the arm length L: from the joint
    axis (where the arm frame is) to the centre of the ball.
    """
    child = link_index(robot, link_name)
    parent = p.getJointInfo(robot, child)[16]
    child_pos = np.array(p.getLinkState(robot, child, computeForwardKinematics=True)[4])
    if parent == -1:
        parent_pos = np.array(p.getBasePositionAndOrientation(robot)[0])
    else:
        parent_pos = np.array(p.getLinkState(robot, parent, computeForwardKinematics=True)[4])
    return float(np.linalg.norm(child_pos - parent_pos))


# ----------------------------------------------------------------------------
# Dynamics model
#
# PyBullet builds the model used by calculateInverseDynamics (and calculateMassMatrix) the first
# time it is called for a robot, and does NOT update it after p.changeDynamics. So we keep a
# hidden copy of each robot in a separate, headless (DIRECT) client, and rebuild it, with the new
# masses set before its first use, every time a mass changes.
# ----------------------------------------------------------------------------
def _close_models():
    for model in _MODELS.values():
        if model["client"] is not None:
            try:
                p.disconnect(physicsClientId=model["client"])
            except Exception:
                pass
    _MODELS.clear()


def _model(robot):
    model = _MODELS[robot]
    if model["client"] is None:
        client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -G, physicsClientId=client)
        body = p.loadURDF(model["urdf"], model["base"], useFixedBase=True,
                          flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
        for link_name, mass in model["masses"].items():
            p.changeDynamics(body, _link_index_in(body, link_name, client), mass=mass,
                             physicsClientId=client)
        model["client"], model["body"] = client, body
    return model


def _link_index_in(body, link_name, client):
    for j in range(p.getNumJoints(body, physicsClientId=client)):
        if p.getJointInfo(body, j, physicsClientId=client)[12].decode() == link_name:
            return j
    raise KeyError(f"no link called '{link_name}'")


def change_mass(robot, link_name, mass, model_mass=None):
    """Change the mass of a link [kg] in the simulated robot AND in its dynamics model.

    model_mass: if given, the model gets this mass instead, so the controller's model is wrong
    (e.g. the real payload is 12 kg but the model believes it is 10 kg).
    """
    p.changeDynamics(robot, link_index(robot, link_name), mass=mass)
    model = _MODELS[robot]
    model["masses"][link_name] = mass if model_mass is None else model_mass
    if model["client"] is not None:            # rebuild the model with the new mass
        p.disconnect(physicsClientId=model["client"])
        model["client"], model["body"] = None, None


def inverse_dynamics(robot, q, qd, qdd):
    """Same as p.calculateInverseDynamics(robot, q, qd, qdd), but always uses the current masses.

    Returns the joint torques M(q) qdd + C(q, qd) qd + g(q) as a list.
    """
    model = _model(robot)
    return p.calculateInverseDynamics(model["body"], list(q), list(qd), list(qdd),
                                      physicsClientId=model["client"])


def mass_matrix(robot, q):
    """Same as p.calculateMassMatrix(robot, q), but always uses the current masses."""
    model = _model(robot)
    return np.array(p.calculateMassMatrix(model["body"], list(q), physicsClientId=model["client"]))


# ----------------------------------------------------------------------------
# Cartesian reference trajectories
# ----------------------------------------------------------------------------
def _smooth_step(x):
    """0 -> 1 as x goes 0 -> 1, with zero speed and acceleration at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def create_triangle_traj(vertices, lap_time, laps=1, dt=DT):
    """Point moving around a triangle at constant speed, starting at vertices[0].

    vertices : 3 points [x, y, z] (world frame, metres)
    lap_time : time for one lap around the triangle [s]
    Returns (times, positions): arrays of shape (N,) and (N, 3), one row every dt.
    """
    corners = np.asarray(vertices, dtype=float)
    corners = np.vstack([corners, corners[:1]])                  # close the triangle
    lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    distance_at_corner = np.concatenate([[0.0], np.cumsum(lengths)])
    perimeter = distance_at_corner[-1]

    times = np.arange(0.0, lap_time * laps, dt)
    distance = (times / lap_time % 1.0) * perimeter               # distance travelled in this lap
    positions = np.column_stack([np.interp(distance, distance_at_corner, corners[:, i]) for i in range(3)])
    return times, positions


def create_3d_helix(start, radius, depth, turns, duration, wait=0.5, dt=DT):
    """Point going down a helix with a vertical axis, starting exactly at `start`.

    The point waits `wait` seconds at `start`, then makes `turns` turns of radius `radius` while
    going down `depth` metres, in `duration` seconds, starting and stopping smoothly.
    Returns (times, positions): arrays of shape (N,) and (N, 3), one row every dt.
    """
    start = np.asarray(start, dtype=float)
    axis_x, axis_y = start[0] - radius, start[1]                  # so that angle 0 is the start
    times = np.arange(0.0, wait + duration + wait, dt)
    s = _smooth_step((times - wait) / duration)
    angle = 2 * np.pi * turns * s
    positions = np.column_stack([axis_x + radius * np.cos(angle),
                                 axis_y + radius * np.sin(angle),
                                 start[2] - depth * s])
    return times, positions


# ----------------------------------------------------------------------------
# Drawing in the 3D window
# ----------------------------------------------------------------------------
def add_marker(position, color=(1, 0.5, 0, 0.8), radius=0.03):
    """Draw a sphere (visual only, no collision, no mass) at `position`. Returns its body id."""
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=list(position))


def draw_polyline(points, color=(0, 0.6, 0), width=2, every=1):
    """Draw a line through `points` (an N x 3 array), using one point out of `every`."""
    if IN_COLAB:
        return
    pts = np.asarray(points)[::every]
    for a, b in zip(pts[:-1], pts[1:]):
        p.addUserDebugLine(a.tolist(), b.tolist(), color, width, 0)


# ----------------------------------------------------------------------------
# Plots (plotly)
# ----------------------------------------------------------------------------
_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]


def plot_series(t, series, ylabel="", title=None, ref=None, ref_label="reference"):
    """Plot several signals against time.

    series : dict {label: 1D array, same length as t}
    ref    : optional reference drawn as a black dashed line (a number, or an array like t)
    """
    t = np.asarray(t)
    if ref is not None and np.ndim(ref) == 0:
        ref = np.full(len(t), float(ref))
    fig = go.Figure()
    for i, (label, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=t, y=np.asarray(y), mode="lines", name=label,
                                 line=dict(color=_COLORS[i % len(_COLORS)])))
    if ref is not None:
        fig.add_trace(go.Scatter(x=t, y=np.asarray(ref), mode="lines", name=ref_label,
                                 line=dict(color="black", dash="dash", width=1)))
    fig.update_layout(title=title, xaxis_title="time [s]", yaxis_title=ylabel, height=350,
                      margin=dict(l=60, r=20, t=50 if title else 20, b=50))
    fig.show()


def plot_joints(t, series, ref=None, joint_names=None, ylabel="angle [rad]", title=None):
    """One panel per joint.

    series : dict {label: N x n array} (one column per joint)
    ref    : optional N x n array, drawn as black dashed lines
    """
    t = np.asarray(t)
    first = np.asarray(next(iter(series.values())))
    n = first.shape[1] if first.ndim > 1 else 1
    names = joint_names or [f"joint {j}" for j in range(n)]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=names,
                        vertical_spacing=0.08)
    for i, (label, y) in enumerate(series.items()):
        y = np.asarray(y).reshape(len(t), n)
        for j in range(n):
            fig.add_trace(go.Scatter(x=t, y=y[:, j], mode="lines", name=label, legendgroup=label,
                                     showlegend=(j == 0), line=dict(color=_COLORS[i % len(_COLORS)])),
                          row=j + 1, col=1)
    if ref is not None:
        ref = np.asarray(ref).reshape(len(t), n)
        for j in range(n):
            fig.add_trace(go.Scatter(x=t, y=ref[:, j], mode="lines", name="reference",
                                     legendgroup="reference", showlegend=(j == 0),
                                     line=dict(color="black", dash="dash", width=1)), row=j + 1, col=1)
    for j in range(n):
        fig.update_yaxes(title_text=ylabel, row=j + 1, col=1)
    fig.update_xaxes(title_text="time [s]", row=n, col=1)
    fig.update_layout(title=title, height=260 * n + 60, margin=dict(l=60, r=20, t=60, b=50))
    fig.show()


def plot_path3d(paths, ref=None, title=None):
    """3D plot of point paths. paths: dict {label: N x 3 array}; ref: optional N x 3 array."""
    fig = go.Figure()
    if ref is not None:
        ref = np.asarray(ref)
        fig.add_trace(go.Scatter3d(x=ref[:, 0], y=ref[:, 1], z=ref[:, 2], mode="lines",
                                   name="reference", line=dict(color="black", dash="dash", width=3)))
    for i, (label, P) in enumerate(paths.items()):
        P = np.asarray(P)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", name=label,
                                   line=dict(color=_COLORS[i % len(_COLORS)], width=4)))
    fig.update_layout(title=title, height=550, scene=dict(aspectmode="data", xaxis_title="x [m]",
                                                          yaxis_title="y [m]", zaxis_title="z [m]"))
    fig.show()
    return fig




def plot_path2d(paths, ref=None, axes=("x", "z"), title=None):
    """2D plot of point paths, seen along the missing axis (default: the x-z plane, seen from y).

    paths: dict {label: N x 3 array}; ref: optional N x 3 array, drawn as a black dashed line.
    axes : the two coordinates to plot, from "x", "y", "z". Equal scale on both axes.
    """
    col = {"x": 0, "y": 1, "z": 2}
    a, b = col[axes[0]], col[axes[1]]
    fig = go.Figure()
    if ref is not None:
        ref = np.asarray(ref)
        fig.add_trace(go.Scatter(x=ref[:, a], y=ref[:, b], mode="lines", name="reference",
                                 line=dict(color="black", dash="dash", width=1.5)))
    for i, (label, P) in enumerate(paths.items()):
        P = np.asarray(P)
        fig.add_trace(go.Scatter(x=P[:, a], y=P[:, b], mode="lines", name=label,
                                 line=dict(color=_COLORS[i % len(_COLORS)], width=2)))
    fig.update_layout(title=title, xaxis_title=f"{axes[0]} [m]", yaxis_title=f"{axes[1]} [m]",
                      height=500, margin=dict(l=60, r=20, t=50 if title else 20, b=50))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()
# ----------------------------------------------------------------------------
# Checking
# ----------------------------------------------------------------------------
def check(label, got, expected, tol=1e-3):
    """Print PASS/FAIL comparing two numbers or arrays."""
    got, expected = np.atleast_1d(np.asarray(got, float)), np.atleast_1d(np.asarray(expected, float))
    err = np.linalg.norm(got - expected)
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {label}: got {np.round(got, 4)}, expected {np.round(expected, 4)}, error {err:.2e}")
    return err < tol
