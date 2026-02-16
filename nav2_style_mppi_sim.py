import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import time

# ============================================================
# NAV2-STYLE MPPI SIM (2D diff-drive-ish kinematic model)
# ============================================================

# ----------------------------
# Simulation / MPPI parameters
# ----------------------------
DT = 0.05
HORIZON = 80               # time steps in rollout
BATCH = 1500               # number of sampled trajectories per iteration
ITERATIONS = 4             # inner-loop MPPI iterations per control cycle (Nav2 often ~2-5)

# Velocity limits
VX_MIN, VX_MAX = 0.0, 1.0
WZ_MIN, WZ_MAX = -1.5, 1.5

# Noise stddev (like nav2 vx_std, wz_std)
VX_STD = 0.25
WZ_STD = 0.50

# MPPI temperature (lambda)
LAMBDA = 1.0

# Low-pass correlation for noise (smooth rollouts)
# Higher alpha -> more correlation across time -> smoother
NOISE_LP_ALPHA = 0.85

# Cost weights (critics)
W_GOAL = 3.0
W_TERMINAL = 12.0
W_OBS = 30.0
W_PATH = 4.0
W_HEADING = 1.5
W_SMOOTH = 0.2
CONTROL_WEIGHT = 0.05

# World/grid
GRID_SIZE = 220
WORLD_SIZE = 16.0
RES = WORLD_SIZE / GRID_SIZE

# Visualization
SHOW_ROLLOUTS = 120        # how many rollouts to plot (subsampled)

# ----------------------------
# Scenario A world
# ----------------------------
goal = np.array([6.0, 0.0], dtype=float)
start_state = np.array([0.0, 0.0, 0.0], dtype=float)  # x, y, theta
obstacle_center = np.array([3.0, 0.0], dtype=float)
obstacle_radius = 0.6

# Path reference: straight line from start to goal
# (Nav2 uses a global plan; here we mimic it with a polyline)
PATH_P0 = np.array([0.0, 0.0], dtype=float)
PATH_P1 = goal.copy()
PATH_DIR = PATH_P1 - PATH_P0
PATH_LEN = np.linalg.norm(PATH_DIR) + 1e-9
PATH_U = PATH_DIR / PATH_LEN  # unit direction

# ----------------------------
# Build occupancy + distance field (costmap proxy)
# ----------------------------
occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
xs = (np.arange(GRID_SIZE) * RES) - WORLD_SIZE / 2
ys = (np.arange(GRID_SIZE) * RES) - WORLD_SIZE / 2
X, Y = np.meshgrid(xs, ys, indexing="ij")
R = np.sqrt((X - obstacle_center[0])**2 + (Y - obstacle_center[1])**2)
occupancy[R < obstacle_radius] = 1

# Distance to obstacle in meters (outside obstacle); inside obstacle distance=0
dist_field = distance_transform_edt(1 - occupancy) * RES

def world_to_grid(x, y):
    ix = ((x + WORLD_SIZE/2) / RES).astype(np.int32)
    iy = ((y + WORLD_SIZE/2) / RES).astype(np.int32)
    return ix, iy

# ----------------------------
# Vectorized rollout dynamics
# ----------------------------
def rollout_batch(state, U_batch):
    """
    state: (3,) x,y,theta
    U_batch: (B, T, 2) controls [vx, wz]
    returns: X: (B, T, 3)
    """
    B, T, _ = U_batch.shape
    Xtraj = np.zeros((B, T, 3), dtype=np.float32)

    x = np.full((B,), state[0], dtype=np.float32)
    y = np.full((B,), state[1], dtype=np.float32)
    th = np.full((B,), state[2], dtype=np.float32)

    vx = np.clip(U_batch[:, :, 0], VX_MIN, VX_MAX).astype(np.float32)
    wz = np.clip(U_batch[:, :, 1], WZ_MIN, WZ_MAX).astype(np.float32)

    for t in range(T):
        x = x + vx[:, t] * np.cos(th) * DT
        y = y + vx[:, t] * np.sin(th) * DT
        th = th + wz[:, t] * DT

        Xtraj[:, t, 0] = x
        Xtraj[:, t, 1] = y
        Xtraj[:, t, 2] = th

    return Xtraj

# ----------------------------
# Critics (cost terms) – vectorized
# ----------------------------
def path_cross_track_cost(x, y):
    """
    Distance from point(s) to the straight path segment PATH_P0->PATH_P1.
    Works for arrays x,y of shape (B,T).
    """
    # projection scalar along path
    px = x - PATH_P0[0]
    py = y - PATH_P0[1]
    s = px * PATH_U[0] + py * PATH_U[1]           # along-track
    s_clamped = np.clip(s, 0.0, PATH_LEN)         # within segment
    projx = PATH_P0[0] + s_clamped * PATH_U[0]
    projy = PATH_P0[1] + s_clamped * PATH_U[1]
    dx = x - projx
    dy = y - projy
    return dx*dx + dy*dy

def heading_to_path_cost(theta):
    """
    Penalize heading away from path direction (like a light PathAngle-ish critic).
    theta: (B,T)
    """
    path_ang = np.arctan2(PATH_U[1], PATH_U[0])
    d = np.arctan2(np.sin(theta - path_ang), np.cos(theta - path_ang))
    return d*d

def obstacle_cost_from_dist(x, y):
    """
    Convert distance-to-obstacle into a soft penalty.
    x,y: (B,T)
    """
    ix, iy = world_to_grid(x, y)
    inside = (ix >= 0) & (ix < GRID_SIZE) & (iy >= 0) & (iy < GRID_SIZE)

    # default huge cost if outside map
    d = np.full_like(x, 0.0, dtype=np.float32)
    d[inside] = dist_field[ix[inside], iy[inside]].astype(np.float32)

    # Penalty: strong when close, saturate inside obstacle
    # This is closer to "costmap critic" feel than singular 1/d^2.
    # d=0 -> exp(0)=1
    # d=0.5 -> exp(-1.5)=0.22
    # Multiply by weight to shape.
    penalty = np.exp(-3.0 * d)

    # Add hard collision penalty (inside obstacle == d ~ 0 and occupancy==1)
    occ = np.zeros_like(ix, dtype=np.float32)
    occ[inside] = occupancy[ix[inside], iy[inside]].astype(np.float32)
    penalty = penalty + 5.0 * occ

    # Outside map -> big
    penalty[~inside] = 50.0
    return penalty

def total_cost(Xtraj, U_batch):
    """
    Xtraj: (B,T,3)
    U_batch: (B,T,2)
    returns: costs (B,)
    """
    x = Xtraj[:, :, 0]
    y = Xtraj[:, :, 1]
    th = Xtraj[:, :, 2]

    # running costs
    goal_dist2 = (x - goal[0])**2 + (y - goal[1])**2
    c_goal = goal_dist2.mean(axis=1)

    c_path = path_cross_track_cost(x, y).mean(axis=1)

    c_head = heading_to_path_cost(th).mean(axis=1)

    c_obs = obstacle_cost_from_dist(x, y).mean(axis=1)

    # control smoothness / effort
    du = np.diff(U_batch, axis=1)
    c_smooth = (du**2).mean(axis=(1, 2))
    c_effort = (U_batch**2).mean(axis=(1, 2))

    # terminal cost
    xt = Xtraj[:, -1, 0]
    yt = Xtraj[:, -1, 1]
    term = (xt - goal[0])**2 + (yt - goal[1])**2

    return (
        W_GOAL * c_goal +
        W_PATH * c_path +
        W_HEADING * c_head +
        W_OBS * c_obs +
        W_SMOOTH * c_smooth +
        CONTROL_WEIGHT * c_effort +
        W_TERMINAL * term
    ).astype(np.float32)

# ----------------------------
# Noise generation (correlated in time)
# ----------------------------
def correlated_noise(batch, horizon, vx_std, wz_std, alpha):
    """
    Returns eps: (B,T,2) with low-pass correlation across time.
    """
    eps = np.random.normal(0.0, 1.0, size=(batch, horizon, 2)).astype(np.float32)
    eps[:, :, 0] *= vx_std
    eps[:, :, 1] *= wz_std

    # low-pass filter each rollout over time (AR(1)-like)
    for t in range(1, horizon):
        eps[:, t, :] = alpha * eps[:, t-1, :] + (1.0 - alpha) * eps[:, t, :]
    return eps

# ----------------------------
# MPPI controller (Nav2-style: iterate, update U, execute first, shift)
# ----------------------------
U = np.zeros((HORIZON, 2), dtype=np.float32)  # persistent control sequence

def mppi_control_step(state):
    global U

    # Warm-start nominal: small forward bias toward goal direction
    goal_angle = np.arctan2(goal[1] - state[1], goal[0] - state[0])
    heading_err = np.arctan2(np.sin(goal_angle - state[2]), np.cos(goal_angle - state[2]))
    U[:, 0] = np.clip(U[:, 0], VX_MIN, VX_MAX)
    U[:, 1] = np.clip(U[:, 1], WZ_MIN, WZ_MAX)

    # If U is near-zero early on, help it start moving
    if np.abs(U[:, 0]).mean() < 0.05:
        U[:, 0] = 0.4
        U[:, 1] = 0.6 * heading_err

    sample_trajs = None
    best_trajs = None

    for it in range(ITERATIONS):
        eps = correlated_noise(BATCH, HORIZON, VX_STD, WZ_STD, NOISE_LP_ALPHA)
        U_batch = U[None, :, :] + eps

        # Apply limits (Nav2 applies constraints to each sampled sequence)
        U_batch[:, :, 0] = np.clip(U_batch[:, :, 0], VX_MIN, VX_MAX)
        U_batch[:, :, 1] = np.clip(U_batch[:, :, 1], WZ_MIN, WZ_MAX)

        Xtraj = rollout_batch(state, U_batch)
        costs = total_cost(Xtraj, U_batch)

        # Softmax weights
        beta = costs.min()
        w = np.exp(-(costs - beta) / LAMBDA).astype(np.float32)
        w = w / (w.sum() + 1e-9)

        # Update U with weighted noise (Nav2 does softmax update of controls)
        dU = (w[:, None, None] * eps).sum(axis=0)
        U = U + dU

        # Keep U feasible
        U[:, 0] = np.clip(U[:, 0], VX_MIN, VX_MAX)
        U[:, 1] = np.clip(U[:, 1], WZ_MIN, WZ_MAX)

        # For visualization: keep last iteration’s rollouts
        if it == ITERATIONS - 1:
            sample_trajs = Xtraj

            # Also keep best few to visualize “fan”
            best_idx = np.argsort(costs)[:SHOW_ROLLOUTS]
            best_trajs = Xtraj[best_idx]

    # Execute first control
    u0 = U[0].copy()

    # Receding horizon shift
    U[:-1] = U[1:]
    U[-1] = U[-2]

    return u0, best_trajs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import time

# ============================================================
# NAV2-STYLE MPPI SIM (2D diff-drive-ish kinematic model)
# ============================================================

# ----------------------------
# Simulation / MPPI parameters
# ----------------------------
DT = 0.05
HORIZON = 80               # time steps in rollout
BATCH = 1500               # number of sampled trajectories per iteration
ITERATIONS = 4             # inner-loop MPPI iterations per control cycle (Nav2 often ~2-5)

# Velocity limits
VX_MIN, VX_MAX = 0.0, 1.0
WZ_MIN, WZ_MAX = -1.5, 1.5

# Noise stddev (like nav2 vx_std, wz_std)
VX_STD = 0.25
WZ_STD = 0.50

# MPPI temperature (lambda)
LAMBDA = 1.0

# Low-pass correlation for noise (smooth rollouts)
# Higher alpha -> more correlation across time -> smoother
NOISE_LP_ALPHA = 0.85

# Cost weights (critics)
W_GOAL = 3.0
W_TERMINAL = 12.0
W_OBS = 30.0
W_PATH = 4.0
W_HEADING = 1.5
W_SMOOTH = 0.2
CONTROL_WEIGHT = 0.05

# World/grid
GRID_SIZE = 220
WORLD_SIZE = 16.0
RES = WORLD_SIZE / GRID_SIZE

# Visualization
SHOW_ROLLOUTS = 120        # how many rollouts to plot (subsampled)

# ----------------------------
# Scenario A world
# ----------------------------
goal = np.array([6.0, 0.0], dtype=float)
start_state = np.array([0.0, 0.0, 0.0], dtype=float)  # x, y, theta
obstacle_center = np.array([3.0, 0.0], dtype=float)
obstacle_radius = 0.6

# Path reference: straight line from start to goal
# (Nav2 uses a global plan; here we mimic it with a polyline)
PATH_P0 = np.array([0.0, 0.0], dtype=float)
PATH_P1 = goal.copy()
PATH_DIR = PATH_P1 - PATH_P0
PATH_LEN = np.linalg.norm(PATH_DIR) + 1e-9
PATH_U = PATH_DIR / PATH_LEN  # unit direction

# ----------------------------
# Build occupancy + distance field (costmap proxy)
# ----------------------------
occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
xs = (np.arange(GRID_SIZE) * RES) - WORLD_SIZE / 2
ys = (np.arange(GRID_SIZE) * RES) - WORLD_SIZE / 2
X, Y = np.meshgrid(xs, ys, indexing="ij")
R = np.sqrt((X - obstacle_center[0])**2 + (Y - obstacle_center[1])**2)
occupancy[R < obstacle_radius] = 1

# Distance to obstacle in meters (outside obstacle); inside obstacle distance=0
dist_field = distance_transform_edt(1 - occupancy) * RES

def world_to_grid(x, y):
    ix = ((x + WORLD_SIZE/2) / RES).astype(np.int32)
    iy = ((y + WORLD_SIZE/2) / RES).astype(np.int32)
    return ix, iy

# ----------------------------
# Vectorized rollout dynamics
# ----------------------------
def rollout_batch(state, U_batch):
    """
    state: (3,) x,y,theta
    U_batch: (B, T, 2) controls [vx, wz]
    returns: X: (B, T, 3)
    """
    B, T, _ = U_batch.shape
    Xtraj = np.zeros((B, T, 3), dtype=np.float32)

    x = np.full((B,), state[0], dtype=np.float32)
    y = np.full((B,), state[1], dtype=np.float32)
    th = np.full((B,), state[2], dtype=np.float32)

    vx = np.clip(U_batch[:, :, 0], VX_MIN, VX_MAX).astype(np.float32)
    wz = np.clip(U_batch[:, :, 1], WZ_MIN, WZ_MAX).astype(np.float32)

    for t in range(T):
        x = x + vx[:, t] * np.cos(th) * DT
        y = y + vx[:, t] * np.sin(th) * DT
        th = th + wz[:, t] * DT

        Xtraj[:, t, 0] = x
        Xtraj[:, t, 1] = y
        Xtraj[:, t, 2] = th

    return Xtraj

# ----------------------------
# Critics (cost terms) – vectorized
# ----------------------------
def path_cross_track_cost(x, y):
    """
    Distance from point(s) to the straight path segment PATH_P0->PATH_P1.
    Works for arrays x,y of shape (B,T).
    """
    # projection scalar along path
    px = x - PATH_P0[0]
    py = y - PATH_P0[1]
    s = px * PATH_U[0] + py * PATH_U[1]           # along-track
    s_clamped = np.clip(s, 0.0, PATH_LEN)         # within segment
    projx = PATH_P0[0] + s_clamped * PATH_U[0]
    projy = PATH_P0[1] + s_clamped * PATH_U[1]
    dx = x - projx
    dy = y - projy
    return dx*dx + dy*dy

def heading_to_path_cost(theta):
    """
    Penalize heading away from path direction (like a light PathAngle-ish critic).
    theta: (B,T)
    """
    path_ang = np.arctan2(PATH_U[1], PATH_U[0])
    d = np.arctan2(np.sin(theta - path_ang), np.cos(theta - path_ang))
    return d*d

def obstacle_cost_from_dist(x, y):
    """
    Convert distance-to-obstacle into a soft penalty.
    x,y: (B,T)
    """
    ix, iy = world_to_grid(x, y)
    inside = (ix >= 0) & (ix < GRID_SIZE) & (iy >= 0) & (iy < GRID_SIZE)

    # default huge cost if outside map
    d = np.full_like(x, 0.0, dtype=np.float32)
    d[inside] = dist_field[ix[inside], iy[inside]].astype(np.float32)

    # Penalty: strong when close, saturate inside obstacle
    # This is closer to "costmap critic" feel than singular 1/d^2.
    # d=0 -> exp(0)=1
    # d=0.5 -> exp(-1.5)=0.22
    # Multiply by weight to shape.
    penalty = np.exp(-3.0 * d)

    # Add hard collision penalty (inside obstacle == d ~ 0 and occupancy==1)
    occ = np.zeros_like(ix, dtype=np.float32)
    occ[inside] = occupancy[ix[inside], iy[inside]].astype(np.float32)
    penalty = penalty + 5.0 * occ

    # Outside map -> big
    penalty[~inside] = 50.0
    return penalty

def total_cost(Xtraj, U_batch):
    """
    Xtraj: (B,T,3)
    U_batch: (B,T,2)
    returns: costs (B,)
    """
    x = Xtraj[:, :, 0]
    y = Xtraj[:, :, 1]
    th = Xtraj[:, :, 2]

    # running costs
    goal_dist2 = (x - goal[0])**2 + (y - goal[1])**2
    c_goal = goal_dist2.mean(axis=1)

    c_path = path_cross_track_cost(x, y).mean(axis=1)

    c_head = heading_to_path_cost(th).mean(axis=1)

    c_obs = obstacle_cost_from_dist(x, y).mean(axis=1)

    # control smoothness / effort
    du = np.diff(U_batch, axis=1)
    c_smooth = (du**2).mean(axis=(1, 2))
    c_effort = (U_batch**2).mean(axis=(1, 2))

    # terminal cost
    xt = Xtraj[:, -1, 0]
    yt = Xtraj[:, -1, 1]
    term = (xt - goal[0])**2 + (yt - goal[1])**2

    return (
        W_GOAL * c_goal +
        W_PATH * c_path +
        W_HEADING * c_head +
        W_OBS * c_obs +
        W_SMOOTH * c_smooth +
        CONTROL_WEIGHT * c_effort +
        W_TERMINAL * term
    ).astype(np.float32)

# ----------------------------
# Noise generation (correlated in time)
# ----------------------------
def correlated_noise(batch, horizon, vx_std, wz_std, alpha):
    """
    Returns eps: (B,T,2) with low-pass correlation across time.
    """
    eps = np.random.normal(0.0, 1.0, size=(batch, horizon, 2)).astype(np.float32)
    eps[:, :, 0] *= vx_std
    eps[:, :, 1] *= wz_std

    # low-pass filter each rollout over time (AR(1)-like)
    for t in range(1, horizon):
        eps[:, t, :] = alpha * eps[:, t-1, :] + (1.0 - alpha) * eps[:, t, :]
    return eps

# ----------------------------
# MPPI controller (Nav2-style: iterate, update U, execute first, shift)
# ----------------------------
U = np.zeros((HORIZON, 2), dtype=np.float32)  # persistent control sequence

def mppi_control_step(state):
    global U

    # Warm-start nominal: small forward bias toward goal direction
    goal_angle = np.arctan2(goal[1] - state[1], goal[0] - state[0])
    heading_err = np.arctan2(np.sin(goal_angle - state[2]), np.cos(goal_angle - state[2]))
    U[:, 0] = np.clip(U[:, 0], VX_MIN, VX_MAX)
    U[:, 1] = np.clip(U[:, 1], WZ_MIN, WZ_MAX)

    # If U is near-zero early on, help it start moving
    if np.abs(U[:, 0]).mean() < 0.05:
        U[:, 0] = 0.4
        U[:, 1] = 0.6 * heading_err

    sample_trajs = None
    best_trajs = None

    for it in range(ITERATIONS):
        eps = correlated_noise(BATCH, HORIZON, VX_STD, WZ_STD, NOISE_LP_ALPHA)
        U_batch = U[None, :, :] + eps

        # Apply limits (Nav2 applies constraints to each sampled sequence)
        U_batch[:, :, 0] = np.clip(U_batch[:, :, 0], VX_MIN, VX_MAX)
        U_batch[:, :, 1] = np.clip(U_batch[:, :, 1], WZ_MIN, WZ_MAX)

        Xtraj = rollout_batch(state, U_batch)
        costs = total_cost(Xtraj, U_batch)

        # Softmax weights
        beta = costs.min()
        w = np.exp(-(costs - beta) / LAMBDA).astype(np.float32)
        w = w / (w.sum() + 1e-9)

        # Update U with weighted noise (Nav2 does softmax update of controls)
        dU = (w[:, None, None] * eps).sum(axis=0)
        U = U + dU

        # Keep U feasible
        U[:, 0] = np.clip(U[:, 0], VX_MIN, VX_MAX)
        U[:, 1] = np.clip(U[:, 1], WZ_MIN, WZ_MAX)

        # For visualization: keep last iteration’s rollouts
        if it == ITERATIONS - 1:
            sample_trajs = Xtraj

            # Also keep best few to visualize “fan”
            best_idx = np.argsort(costs)[:SHOW_ROLLOUTS]
            best_trajs = Xtraj[best_idx]

    # Execute first control
    u0 = U[0].copy()

    # Receding horizon shift
    U[:-1] = U[1:]
    U[-1] = U[-2]

    return u0, best_trajs

# ----------------------------
# Main simulation loop (real-time plotting)
# ----------------------------
# state = start_state.copy()
# traj_exec = [state.copy()]

# plt.ion()
# fig, ax = plt.subplots()

# for step in range(500):
#     t0 = time.time()

#     dist_to_goal = np.linalg.norm(state[:2] - goal)
#     if dist_to_goal < 0.25:
#         u = np.array([0.0, 0.0], dtype=np.float32)
#         best_trajs = None
#     else:
#         u, best_trajs = mppi_control_step(state)

#     # Apply control
#     vx = float(np.clip(u[0], VX_MIN, VX_MAX))
#     wz = float(np.clip(u[1], WZ_MIN, WZ_MAX))

#     state[0] += vx * np.cos(state[2]) * DT
#     state[1] += vx * np.sin(state[2]) * DT
#     state[2] += wz * DT

#     traj_exec.append(state.copy())

#     # Plot
#     ax.clear()

#     # Best rollouts (subsample)
#     if best_trajs is not None:
#         for i in range(min(best_trajs.shape[0], SHOW_ROLLOUTS)):
#             ax.plot(best_trajs[i, :, 0], best_trajs[i, :, 1], color="gray", alpha=0.12)

#     traj_np = np.array(traj_exec)
#     ax.plot(traj_np[:, 0], traj_np[:, 1], "b", linewidth=2)

#     ax.scatter(goal[0], goal[1], c="green", s=100, label="Goal")
#     ax.add_patch(plt.Circle(obstacle_center, obstacle_radius, color="red"))

#     # draw path (global plan proxy)
#     ax.plot([PATH_P0[0], PATH_P1[0]], [PATH_P0[1], PATH_P1[1]], "--", linewidth=1)

#     ax.set_xlim(-2, 8)
#     ax.set_ylim(-4, 4)
#     ax.set_aspect("equal")

#     hz = 1.0 / max(1e-6, (time.time() - t0))
#     ax.set_title(f"Nav2-style MPPI | Iter={ITERATIONS} Batch={BATCH} | {hz:.2f} Hz")

#     plt.pause(0.001)

# plt.ioff()
# plt.show()

# ----------------------------
# Main simulation loop (real-time plotting)
# ----------------------------
# state = start_state.copy()
# traj_exec = [state.copy()]

# plt.ion()
# fig, ax = plt.subplots()

# for step in range(500):
#     t0 = time.time()

#     dist_to_goal = np.linalg.norm(state[:2] - goal)
#     if dist_to_goal < 0.25:
#         u = np.array([0.0, 0.0], dtype=np.float32)
#         best_trajs = None
#     else:
#         u, best_trajs = mppi_control_step(state)

#     # Apply control
#     vx = float(np.clip(u[0], VX_MIN, VX_MAX))
#     wz = float(np.clip(u[1], WZ_MIN, WZ_MAX))

#     state[0] += vx * np.cos(state[2]) * DT
#     state[1] += vx * np.sin(state[2]) * DT
#     state[2] += wz * DT

#     traj_exec.append(state.copy())

#     # Plot
#     ax.clear()

#     # Best rollouts (subsample)
#     if best_trajs is not None:
#         for i in range(min(best_trajs.shape[0], SHOW_ROLLOUTS)):
#             ax.plot(best_trajs[i, :, 0], best_trajs[i, :, 1], color="gray", alpha=0.12)

#     traj_np = np.array(traj_exec)
#     ax.plot(traj_np[:, 0], traj_np[:, 1], "b", linewidth=2)

#     ax.scatter(goal[0], goal[1], c="green", s=100, label="Goal")
#     ax.add_patch(plt.Circle(obstacle_center, obstacle_radius, color="red"))

#     # draw path (global plan proxy)
#     ax.plot([PATH_P0[0], PATH_P1[0]], [PATH_P0[1], PATH_P1[1]], "--", linewidth=1)

#     ax.set_xlim(-2, 8)
#     ax.set_ylim(-4, 4)
#     ax.set_aspect("equal")

#     hz = 1.0 / max(1e-6, (time.time() - t0))
#     ax.set_title(f"Nav2-style MPPI | Iter={ITERATIONS} Batch={BATCH} | {hz:.2f} Hz")

#     plt.pause(0.001)

# plt.ioff()
# plt.show()
def run_single_sim(seed=None, render=False):

    global U
    U = np.zeros((HORIZON, 2), dtype=np.float32)

    if seed is not None:
        np.random.seed(seed)

    state = start_state.copy()
    min_clearance = 1e9
    collided = False
    trajectory = [state.copy()]

    if render:
        plt.ion()
        fig, ax = plt.subplots()

    for step in range(700):  # 35 seconds max

        dist_to_goal = np.linalg.norm(state[:2] - goal)

        # Stop if close enough
        if dist_to_goal < 0.3:
            break

        u, best_trajs = mppi_control_step(state)
            

        vx = float(np.clip(u[0], VX_MIN, VX_MAX))
        wz = float(np.clip(u[1], WZ_MIN, WZ_MAX))

        state[0] += vx * np.cos(state[2]) * DT
        state[1] += vx * np.sin(state[2]) * DT
        state[2] += wz * DT

        trajectory.append(state.copy())

        ix, iy = world_to_grid(state[0], state[1])

        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
            d = dist_field[ix, iy]
            min_clearance = min(min_clearance, d)
            if occupancy[ix, iy] == 1:
                collided = True
                break
        else:
            collided = True
            break

        # -------------------------
        # REAL-TIME RENDERING
        # -------------------------
        if render:
            ax.clear()

            if best_trajs is not None:
                for i in range(min(best_trajs.shape[0], SHOW_ROLLOUTS)):
                    ax.plot(best_trajs[i, :, 0],
                            best_trajs[i, :, 1],
                            color="gray", alpha=0.12)

            traj_np = np.array(trajectory)
            ax.plot(traj_np[:, 0], traj_np[:, 1], "b", linewidth=2)

            ax.scatter(goal[0], goal[1], c="green", s=100)
            ax.add_patch(plt.Circle(obstacle_center,
                                    obstacle_radius,
                                    color="red"))

            ax.plot([PATH_P0[0], PATH_P1[0]],
                    [PATH_P0[1], PATH_P1[1]],
                    "--", linewidth=1)

            ax.set_xlim(-2, 8)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.set_title("Nav2-style MPPI (Render Mode)")

            plt.pause(0.001)

    if render:
        ax.set_title("Reached Goal - Press Close Window")
        plt.pause(0.1)
        plt.ioff()
        plt.show()


    reached = np.linalg.norm(state[:2] - goal) < 0.3
    time_taken = step * DT

    success = (
        reached and
        not collided and
        min_clearance > 0.05 and
        time_taken < 30.0
    )

    return {
        "success": success,
        "reached": reached,
        "collided": collided,
        "min_clearance": float(min_clearance),
        "time": float(time_taken)
    }

import json
from datetime import datetime

def run_batch(N=100):

    results = []
    failed_seeds = []
    successes = 0

    print(f"\nRunning batch of {N} simulations...\n")

    for seed in range(N):

        r = run_single_sim(seed=seed, render=False)
        r["seed"] = seed
        results.append(r)

        if r["success"]:
            successes += 1
        else:
            failed_seeds.append(seed)

        running_rate = successes / (seed + 1)

        print(
            f"Seed {seed:3d} | "
            f"Success: {r['success']} | "
            f"Time: {r['time']:.2f}s | "
            f"Clearance: {r['min_clearance']:.3f} | "
            f"Running Success Rate: {running_rate:.3f}"
        )

    final_rate = successes / N

    print("\n========================")
    print(f"FINAL SUCCESS RATE: {final_rate:.3f}")
    print(f"Total Failures: {len(failed_seeds)}")
    print("Failed Seeds:", failed_seeds)
    print("========================\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mppi_batch_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump({
            "success_rate": final_rate,
            "failed_seeds": failed_seeds,
            "results": results
        }, f, indent=4)

    print(f"Saved results to {filename}")

    return filename


def replay_from_file(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    failed = data["failed_seeds"]

    print("Failed seeds:", failed)

    for seed in failed:
        print(f"\nReplaying seed {seed}")
        run_single_sim(seed=seed, render=True)

if __name__ == "__main__":
    run_single_sim(render=True)
    # result_file = run_batch(N=100)

    # print("\nTo replay failures later:")
    # print(f"replay_from_file('{result_file}')")

