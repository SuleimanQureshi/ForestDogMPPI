"""
Demo 02: Trot forward
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
from dataclasses import dataclass, field

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import Gait
from convex_mpc.plot_helper import plot_mpc_result, plot_swing_foot_traj, plot_solve_time, hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

# Simulation Setting
INITIAL_X_POS = -2
INITIAL_Y_POS = 0
# How long does the simulation run for How much time 
RUN_SIM_LENGTH_S = 15.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ
REALTIME_FACTOR = 1

# Locomotion Command
@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float

# Command format:
#   [start_time (s), end_time (s), x_velocity (m/s), y_velocity (m/s),
#    z_position (m), yaw_angular_velocity (rad/s)]
CMD_SCHEDULE = [
BodyCmdPhase(0.0, 5.0,  0.8, 0.0, 0.27, 0.0),   # Forward 0.8 m/s
]

# Gait Setting
GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

# Trajectory Reference Setting (defaults)
x_vel_des_body = 0.0
y_vel_des_body = 0.0
z_pos_des_body = 0.27
yaw_rate_des_body = 0.0

#MuJoCo Sim Update Rate
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

#Leg Coontroller Update Rate
CTRL_HZ = 200       # 200 Hz
CTRL_DT = 1.0 / CTRL_HZ

# Must be an integer ratio for clean decimation
if SIM_HZ % CTRL_HZ != 0:
    raise ValueError(
        f"SIM_HZ ({SIM_HZ}) must be divisible by CTRL_HZ ({CTRL_HZ}) for this decimation method."
    )
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

# Relation between MPC loop and control loop
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))  # MPC update every N control ticks

# Go2 Joint Torque Limit
HIP_LIM = 23.7
ABD_LIM = 23.7
KNEE_LIM = 45.43
SAFETY = 0.9

TAU_LIM = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FL: hip, thigh, calf
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FR
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RL
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RR
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

# --------------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------------
def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, 0.27, 0.0

def visualize_lidar_hits(viewer, hits_world, max_points=200):
    viewer.user_scn.ngeom = 0  # clear previous frame markers

    if hits_world.shape[0] == 0:
        return

    pts = hits_world[:max_points]

    for p in pts:
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            g,
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=p,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1]   # RED
        )
        viewer.user_scn.ngeom += 1

@dataclass
class GlobalHeightMap:
    size_xy: float = 12.0
    res: float = 0.02

    def __post_init__(self):
        self.N = int(self.size_xy / self.res)
        self.hmap = np.full((self.N, self.N), np.nan, dtype=np.float32)
        self.origin_xy = np.array([-self.size_xy / 2.0, -self.size_xy / 2.0])

    def update(self, hits_world: np.ndarray):
        # DO NOT clear every frame (persistence is the point)
        if hits_world.shape[0] == 0:
            return

        x = hits_world[:, 0]
        y = hits_world[:, 1]
        z = hits_world[:, 2]

        ix = ((x - self.origin_xy[0]) / self.res).astype(int)
        iy = ((y - self.origin_xy[1]) / self.res).astype(int)

        valid = (ix >= 0) & (ix < self.N) & (iy >= 0) & (iy < self.N)
        ix = ix[valid]
        iy = iy[valid]
        z = z[valid]

        # keep max height per cell (helps obstacles remain visible)
        for i, j, zz in zip(ix, iy, z):
            prev = self.hmap[j, i]
            if np.isnan(prev):
                self.hmap[j, i] = zz
            else:
                self.hmap[j, i] = max(prev, zz)

    def query_height_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = ((x - self.origin_xy[0]) / self.res).astype(int)
        iy = ((y - self.origin_xy[1]) / self.res).astype(int)

        valid = (ix >= 0) & (ix < self.N) & (iy >= 0) & (iy < self.N)

        z = np.zeros_like(x, dtype=float)
        vals = self.hmap[iy[valid], ix[valid]]
        vals = np.where(np.isnan(vals), 0.0, vals)
        z[valid] = vals
        return z

    def height_and_normal(self, x: float, y: float):
        """
        Query height map at (x,y).
        Returns:
            z (float)
            normal (3,) numpy array
        """

        ix = int((x - self.origin_xy[0]) / self.res)
        iy = int((y - self.origin_xy[1]) / self.res)

        if ix < 0 or ix >= self.N or iy < 0 or iy >= self.N:
            return 0.0, np.array([0.0, 0.0, 1.0])

        z = self.hmap[iy, ix]

        if np.isnan(z):
            return 0.0, np.array([0.0, 0.0, 1.0])

        # Simple normal estimate using central differences
        dzdx = 0.0
        dzdy = 0.0

        if 1 <= ix < self.N-1 and 1 <= iy < self.N-1:
            z_x1 = self.hmap[iy, ix+1]
            z_x0 = self.hmap[iy, ix-1]
            z_y1 = self.hmap[iy+1, ix]
            z_y0 = self.hmap[iy-1, ix]

            if not np.isnan(z_x1) and not np.isnan(z_x0):
                dzdx = (z_x1 - z_x0) / (2*self.res)

            if not np.isnan(z_y1) and not np.isnan(z_y0):
                dzdy = (z_y1 - z_y0) / (2*self.res)

        normal = np.array([-dzdx, -dzdy, 1.0])
        normal /= (np.linalg.norm(normal) + 1e-9)

        return float(z), normal
class ObstacleCostMap2D:

    def __init__(self, size_xy=12.0, res=0.05):
        self.size_xy = size_xy
        self.res = res
        self.N = int(size_xy / res)

        self.origin_xy = np.array([-size_xy/2, -size_xy/2])
        self.grid = np.zeros((self.N, self.N), dtype=np.float32)

        self.decay = 0.98
        self.inflate_radius = 0.3

    def world_to_grid(self, x, y):
        ix = ((x - self.origin_xy[0]) / self.res).astype(int)
        iy = ((y - self.origin_xy[1]) / self.res).astype(int)
        return ix, iy

    def update(self, hits_world):

        # decay persistent cost field
        self.grid *= self.decay

        if hits_world.shape[0] == 0:
            return

        # --- Create temporary raw obstacle grid ---
        raw = np.zeros_like(self.grid)

        x = hits_world[:,0]
        y = hits_world[:,1]

        ix, iy = self.world_to_grid(x,y)

        valid = (ix>=0)&(ix<self.N)&(iy>=0)&(iy<self.N)
        ix = ix[valid]
        iy = iy[valid]

        raw[iy, ix] = 1.0

        # --- Inflate raw obstacles ONLY ---
        inflated = self.inflate_from_raw(raw)

        # --- Merge with persistent grid ---
        self.grid = np.maximum(self.grid, inflated)

    def inflate_from_raw(self, raw):

        inflated = raw.copy()
        radius_cells = int(self.inflate_radius / self.res)

        for i in range(self.N):
            for j in range(self.N):
                if raw[j,i] > 0.5:

                    for dx in range(-radius_cells, radius_cells+1):
                        for dy in range(-radius_cells, radius_cells+1):

                            ni = i+dx
                            nj = j+dy

                            if 0<=ni<self.N and 0<=nj<self.N:
                                dist = np.sqrt(dx*dx+dy*dy)*self.res
                                cost = max(0.0, 1.0 - dist/self.inflate_radius)
                                inflated[nj,ni] = max(inflated[nj,ni], cost)

        return inflated

    def query_cost_batch(self, x, y):
        ix, iy = self.world_to_grid(x,y)

        valid = (ix>=0)&(ix<self.N)&(iy>=0)&(iy<self.N)

        cost = np.zeros_like(x)
        cost[valid] = self.grid[iy[valid], ix[valid]]
        return cost



class MuJoCoLidar3D:
    """
    Very simple 3D LiDAR using MuJoCo ray casts.
    Returns hit points in WORLD frame.
    """
    def __init__(self, model, data,
                 n_az=72, n_el=9,
                 el_min_deg=-45.0, el_max_deg=5.0,
                 max_range=6.0):
        self.model = model
        self.data = data
        self.max_range = float(max_range)

        az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
        el = np.linspace(np.deg2rad(el_min_deg), np.deg2rad(el_max_deg), n_el)

        self.dirs_body = []
        for e in el:
            ce, se = np.cos(e), np.sin(e)
            for a in az:
                ca, sa = np.cos(a), np.sin(a)
                # body-frame ray direction (x forward, y left, z up)
                d = np.array([ce * ca, ce * sa, se], dtype=float)
                d /= (np.linalg.norm(d) + 1e-12)
                self.dirs_body.append(d)
        self.dirs_body = np.asarray(self.dirs_body)  # (M,3)

    def scan(self, sensor_pos_world, R_world_from_body, bodyexclude = 1):

        dirs_world = (R_world_from_body @ self.dirs_body.T).T

        hits = []
        p0 = np.asarray(sensor_pos_world, dtype=float)

        geomgroup = np.ones(6, dtype=np.uint8)

        for d in dirs_world:
            geomid = np.array([-1], dtype=np.int32)

            dist = mj.mj_ray(
                self.model,
                self.data,
                p0,
                d,
                None,
                1,
                -1,
                geomid
            )

            if geomid[0] != -1:
                body_id = self.model.geom_bodyid[geomid[0]]
                body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)

                if "go2" in body_name or "trunk" in body_name:
                    continue   # skip robot hits

            if dist > 0 and dist < self.max_range:
                hits.append(p0 + dist * d)

        if len(hits) == 0:
            return np.zeros((0, 3), dtype=float)

        return np.asarray(hits, dtype=float)

class MPPIConfig:
    horizon_steps: int
    dt: float
    num_samples: int = 128
    lambda_: float = 1.0
    noise_std: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.2, 0.6]))
    u_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -0.5, -1.5]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([ 1.0,  0.5,  1.5]))
    w_goal: float = 5.0
    w_u: float = 0.2
    w_obs: float = 200.0
    obs_margin: float = 0.25
class Nav2StyleMPPI:

    def __init__(self, dt):

        # --- MPPI parameters ---
        self.dt = dt
        self.H = 130              # ~1.66s lookahead with dt=0.0208
        self.BATCH = 600         # still feasible
        self.ITERS = 2           # better convergence

        self.LAMBDA = 8.0
        self.ALPHA = 0.1        # correlated noise

        # velocity limits (keep conservative!)
        self.vx_min, self.vx_max = 0.0, 0.8
        self.vy_min, self.vy_max = -0.6, 0.6
        self.wz_min, self.wz_max = -1.5, 1.5
        
        self.costmap = None

        self.best_traj = np.zeros((self.H,3))
        # noise std
        self.std = np.array([0.06, 0.12, 0.12]) # allow lateral/turn exploration

        self.side_bias = 0.0

        # persistent sequence
        self.U = np.zeros((self.H, 3))

        self.terrain = None
    def set_costmap(self, costmap):
        self.costmap = costmap
    def set_terrain(self, heightmap):
        self.terrain = heightmap


    # --------------------------------------
    # correlated noise
    # --------------------------------------
    def correlated_noise(self):
        eps = np.random.randn(self.BATCH, self.H, 3)
        eps *= self.std

        for t in range(1, self.H):
            eps[:, t, :] = (
                self.ALPHA * eps[:, t-1, :]
                + (1.0 - self.ALPHA) * eps[:, t, :]
            )
        # eps[:, :, 0] += 0.05   # small forward bias
        eps[:, :, 1] += self.side_bias

        return eps

    # --------------------------------------
    # rollout model
    # --------------------------------------
    def rollout(self, state, U_batch):

        B, T, _ = U_batch.shape
        X = np.zeros((B, T, 4))

        x = np.full(B, state[0])
        y = np.full(B, state[1])
        yaw = np.full(B, state[2])

        for t in range(T):

            vx = np.clip(U_batch[:, t, 0], self.vx_min, self.vx_max)
            vy = np.clip(U_batch[:, t, 1], self.vy_min, self.vy_max)
            wz = np.clip(U_batch[:, t, 2], self.wz_min, self.wz_max)

            x += (vx*np.cos(yaw) - vy*np.sin(yaw)) * self.dt
            y += (vx*np.sin(yaw) + vy*np.cos(yaw)) * self.dt
            yaw += wz * self.dt

            if self.terrain is not None:
                z = self.terrain.query_height_batch(x, y) + 0.27
            else:
                z = np.full_like(x, 0.27)

            X[:, t, 0] = x
            X[:, t, 1] = y
            X[:, t, 2] = z
            X[:, t, 3] = yaw


        return X

    # --------------------------------------
    # cost
    # --------------------------------------
    def cost(self, X, U_batch, goal, obstacle_xy):


        x = X[:, :, 0]
        y = X[:, :, 1]
        z = X[:, :, 2]
        yaw = X[:, :, 3]


        # goal distance
        goal_cost = ((x - goal[0])**2 + (y - goal[1])**2).mean(axis=1)

        # terminal cost
        term = (x[:, -1] - goal[0])**2 + (y[:, -1] - goal[1])**2

        # heading to goal
        goal_ang = np.arctan2(goal[1] - y, goal[0] - x)
        dtheta = np.arctan2(
            np.sin(yaw - goal_ang),
            np.cos(yaw - goal_ang)
        )
        heading_cost = (dtheta**2).mean(axis=1)

        # obstacle cost (soft + hard safety)
        # obstacle cost from point cloud (no single radius / supports many objects)
        if self.costmap is not None:

            cost_vals = self.costmap.query_cost_batch(
                x.reshape(-1),
                y.reshape(-1)
            ).reshape(x.shape)

            obs_cost = 150.0 * cost_vals.mean(axis=1)

        else:
            obs_cost = np.zeros(x.shape[0])



        # -----------------------------
        # Terrain clearance term
        # -----------------------------
        if self.terrain is not None:
            z_ground = self.terrain.query_height_batch(
                x.reshape(-1), y.reshape(-1)
            ).reshape(x.shape)

            clearance = z - z_ground
            min_clearance = 0.20
            viol = np.maximum(0.0, min_clearance - clearance)
            terrain_cost = 70.0 * (viol**2).mean(axis=1)

            # -----------------------------
            # Slope term (terrain gradient)
            # -----------------------------
            delta = self.terrain.res
            z_x1 = self.terrain.query_height_batch((x + delta).reshape(-1), y.reshape(-1)).reshape(x.shape)
            z_x2 = self.terrain.query_height_batch((x - delta).reshape(-1), y.reshape(-1)).reshape(x.shape)
            z_y1 = self.terrain.query_height_batch(x.reshape(-1), (y + delta).reshape(-1)).reshape(x.shape)
            z_y2 = self.terrain.query_height_batch(x.reshape(-1), (y - delta).reshape(-1)).reshape(x.shape)

            dzdx = (z_x1 - z_x2) / (2 * delta)
            dzdy = (z_y1 - z_y2) / (2 * delta)

            slope_mag = np.sqrt(dzdx**2 + dzdy**2)
            slope_cost = 5.0 * (slope_mag**2).mean(axis=1)
        else:
            terrain_cost = np.zeros(x.shape[0])
            slope_cost = np.zeros(x.shape[0])


        obs_total = obs_cost 

        #Progress param
        d0 = np.sqrt((x[:, 0] - goal[0])**2 + (y[:, 0] - goal[1])**2)      # distance at start of rollout
        dT = np.sqrt((x[:, -1] - goal[0])**2 + (y[:, -1] - goal[1])**2)    # distance at end
        progress = (d0 - dT)  # positive = moved toward goal


        # smoothness
        dU = np.diff(U_batch, axis=1)
        smooth = (dU**2).mean(axis=(1,2))

        # control effort
        effort = (U_batch**2).mean(axis=(1,2))

        # Penalize lateral motion (prefer forward motion)
        lateral_cost = (U_batch[:,:,1]**2).mean(axis=1)

        # Prefer velocity aligned with heading
        vx = U_batch[:,:,0]
        vy = U_batch[:,:,1]

        speed = np.sqrt(vx**2 + vy**2) + 1e-6
        forward_ratio = vx / speed

        # penalize sideways ratio
        direction_cost = ((1.0 - forward_ratio)**2).mean(axis=1)


        # change behaviours near goal
        dist_to_goal = np.sqrt((x[:,0] - goal[0])**2 + (y[:,0] - goal[1])**2)
        near_goal = dist_to_goal < 0.6

        w_goal = 2.0
        w_term = 6.0
        w_heading = 3.0
        w_progress = -2.5

        # amplify terminal behavior near goal
        # w_term = np.where(near_goal, 15.0, w_term)
        # w_heading = np.where(near_goal, 6.0, w_heading)
        # w_progress = np.where(near_goal, -5.0, w_progress)

        return (
            w_goal * goal_cost +
            w_term * term +
            w_heading * heading_cost +
            1.8 * obs_total +
            0.2 * smooth +
            0.05 * effort +
            0.8 * lateral_cost +
            1.5 * direction_cost +
            w_progress * progress
        )





    # --------------------------------------
    # main step
    # --------------------------------------
    def command(self, state, goal, obstacle_xy):
        
        for _ in range(self.ITERS):

            eps = self.correlated_noise()
            U_batch = self.best_traj[None,:,:] + eps
            # Inject symmetric lateral bias
            # lateral_bias = np.linspace(-0.5, 0.5, self.BATCH)
            # U_batch[:, :, 1] += lateral_bias[:, None]

            X = self.rollout(state, U_batch)
            costs = self.cost(X, U_batch, goal, obstacle_xy)

            best_idx = np.argmin(costs)
            best_traj = U_batch[best_idx].copy()
            self.best_traj = best_traj
            # --- Select elite subset ---
            elite_frac = 0.2      # top 20%
            K = int(elite_frac * self.BATCH)
            elite_idx = np.argsort(costs)[:K]

            costs_elite = costs[elite_idx]
            U_elite = U_batch[elite_idx]

            # --- Temperature parameter ---
            tau = 0.4   # tuning knob (try 1.0–5.0)

            beta = costs_elite.min()
            w = np.exp(-(costs_elite - beta) / tau)
            w /= (w.sum() + 1e-9)

            # --- Weighted mean of elite trajectories ---
            self.U = np.sum(w[:, None, None] * U_elite, axis=0)


            # clip feasible
            self.U[:, 0] = np.clip(self.U[:, 0], self.vx_min, self.vx_max)
            self.U[:, 1] = np.clip(self.U[:, 1], self.vy_min, self.vy_max)
            self.U[:, 2] = np.clip(self.U[:, 2], self.wz_min, self.wz_max)

        # execute first control
        u0 = self.U[0].copy()

        # If obstacle present, determine relative lateral offset
        # if obstacle_xy is not None and obstacle_xy.shape[0] > 0:
        #     mean_y = obstacle_xy[:,1].mean()
        #     self.side_bias = -0.2 if mean_y > 0 else 0.2


        # receding horizon
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]

        self.last_U_batch = U_batch
        self.best_traj[:-1] = self.best_traj[1:]
        self.best_traj[-1] = self.best_traj[-2]

        return u0


import matplotlib.pyplot as plt

def debug_plot_mppi(state, goal, obstacle_xy, U_batch, mppi):

    plt.clf()

    # -----------------------------
    # 1️⃣ Draw Costmap
    # -----------------------------
    if mppi.costmap is not None:

        grid = mppi.costmap.grid
        res = mppi.costmap.res
        origin = mppi.costmap.origin_xy

        extent = [
            origin[0],
            origin[0] + grid.shape[1] * res,
            origin[1],
            origin[1] + grid.shape[0] * res,
        ]

        plt.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="hot",
            alpha=0.5,
            vmin=0,
            vmax=1
        )

    # -----------------------------
    # 2️⃣ Rollouts
    # -----------------------------
    if U_batch is not None:
        X = mppi.rollout(state, U_batch)

        for i in range(min(120, X.shape[0])):
            plt.plot(
                X[i,:,0],
                X[i,:,1],
                color='blue',
                alpha=0.15
            )

    # -----------------------------
    # 3️⃣ Robot + Goal
    # -----------------------------
    plt.scatter(state[0], state[1], c='black', s=80, label="Robot")
    plt.scatter(goal[0], goal[1], c='green', s=120, label="Goal")

    # -----------------------------
    # 4️⃣ Raw LiDAR points (optional)
    # -----------------------------
    if obstacle_xy is not None and obstacle_xy.shape[0] > 0:
        plt.scatter(obstacle_xy[:,0], obstacle_xy[:,1],
                    c='cyan', s=5, label="LiDAR")

    plt.axis("equal")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend(loc="upper right")
    plt.pause(0.001)

# --------------------------------------------------------------------------------
# Storage Variables (CONTROL-rate logs for plots)
# --------------------------------------------------------------------------------

# Centroidal state x = [px, py, pz, r, p, y, vx, vy, vz, wx, wy, wz]
x_vec = np.zeros((12, CTRL_STEPS))

# MPC contact force log (world): [FLx,FLy,FLz, FRx,FRy,FRz, RLx,RLy,RLz, RRx,RRy,RRz]
mpc_force_world = np.zeros((12, CTRL_STEPS))

# Torques
tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))

# Control-rate log (if you want it)
time_log_ctrl_s = np.zeros(CTRL_STEPS)
q_log_ctrl = np.zeros((CTRL_STEPS, 19))
tau_log_ctrl_Nm = np.zeros((CTRL_STEPS, 12))

# Foot trajectory logs (control-rate)
@dataclass
class FootTraj:
    pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))


foot_traj = FootTraj()

mpc_update_time_ms = []
mpc_solve_time_ms = []
X_opt = None
U_opt = None
def debug_plot_heightmap(hmap, res, origin):
    plt.figure("Global Height Map", clear=True)
    extent = [
        origin[0],
        origin[0] + hmap.shape[1]*res,
        origin[1],
        origin[1] + hmap.shape[0]*res
    ]
    plt.imshow(heightmap.hmap, origin="lower", extent=extent, vmin=0, vmax=0.5)
    plt.colorbar(label="World Z (m)")
    plt.pause(0.001)
# --------------------------------------------------------------------------------
# Simulation Initialization
# --------------------------------------------------------------------------------

go2 = PinGo2Model()
mujoco_go2 = MuJoCo_GO2_Model()
lidar = MuJoCoLidar3D(
    mujoco_go2.model,
    mujoco_go2.data,
    n_az=90,
    n_el=15,
    el_min_deg=-30.0,
    el_max_deg=15.0,
    max_range=6.0
)
heightmap = GlobalHeightMap(size_xy=12.0, res=0.02)
costmap = ObstacleCostMap2D(size_xy=12.0, res=0.05)
leg_controller = LegController()
traj = ComTraj(go2)
gait = Gait(GAIT_HZ, GAIT_DUTY)

traj.generate_traj(
    go2,
    gait,
    0.0,
    0.0,   # vx
    0.0,   # vy
    0.27,  # z
    0.0,   # yaw rate
    time_step=MPC_DT,
)
mpc = CentroidalMPC(go2, traj)

# mppi_cfg = MPPIConfig(horizon_steps=traj.N, dt=MPC_DT)
mppi = Nav2StyleMPPI(MPC_DT)
mppi.set_terrain(heightmap)
mppi.set_costmap(costmap)
goal_xy = np.array([3.0, 0.0])
box_radius = 0.75

# Initialize robot configuration
q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_go2.update_with_q_pin(q_init)

# Set physics dt (keep it fast!)
mujoco_go2.model.opt.timestep = SIM_DT


# Safe defaults until first solve
obstacle_xy = np.zeros((0, 2), dtype=float)
u0 = np.array([0.0, 0.0, 0.0], dtype=float)
U_opt = np.zeros((12, traj.N), dtype=float)

# --------------------------------------------------------------------------------
# Replay logs sampled at RENDER_HZ
# --------------------------------------------------------------------------------
time_log_render = []
q_log_render = []
tau_log_render = []

next_render_t = 0.0

# --------------------------------------------------------------------------------
# Simulation Loop
# --------------------------------------------------------------------------------
print(f"Running simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)
debug_frames = []
with mj.viewer.launch_passive(mujoco_go2.model, mujoco_go2.data) as viewer:

    viewer.cam.distance = 3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)
        box_id = mj.mj_name2id(
            mujoco_go2.model,
            mj.mjtObj.mjOBJ_BODY,
            "box"
        )

        box_pos = mujoco_go2.data.xpos[box_id].copy()
        


        # Control update at CTRL_HZ
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            # Commands (updated at control rate)
            # x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)

            # Update Pinocchio from current MuJoCo state
            mujoco_go2.update_pin_with_mujoco(go2)

            x_vec[:, ctrl_i] = go2.compute_com_x_vec().reshape(-1)
            px = x_vec[0, ctrl_i]
            py = x_vec[1, ctrl_i]
            yaw = x_vec[5, ctrl_i]
            robot_xy = x_vec[0:2, ctrl_i]
            # box_xy = box_pos[0:2]

            # dist = np.linalg.norm(robot_xy - box_xy)
            # if ctrl_i % 20 == 0:
            #     print("Distance to obstacle:", dist)
            #     print("Distance to goal:", np.linalg.norm(robot_xy - goal_xy))

            # Control-rate logs
            time_log_ctrl_s[ctrl_i] = time_now_s
            q_log_ctrl[ctrl_i, :] = mujoco_go2.data.qpos

            # Update MPC if needed
            if (ctrl_i %( 1 * STEPS_PER_MPC)) == 0:
                print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)
                # --- Provide desired velocities to gait/leg controller (used for swing touchdown prediction) ---
                # MPPI u is in BODY frame (vx, vy, wz) in our planner
                

                state0 = np.array([px, py, yaw])
                state0 = np.array([px, py, yaw])

                if (ctrl_i % (4 * STEPS_PER_MPC)) == 0:
                    # LiDAR origin (a bit above COM so it doesn't hit the ground instantly)
                    lidar_origin = go2.current_config.base_pos.copy()
                    lidar_origin[2] -= 0.1
                    
                    # print("LiDAR origin:", lidar_origin)


                    base_body_id = mj.mj_name2id(
                        mujoco_go2.model,
                        mj.mjtObj.mjOBJ_BODY,
                        "trunk"
                    )

                    Rwb = mujoco_go2.data.xmat[base_body_id].reshape(3,3).copy()

                    # Perform scan using full base rotation
                    trunk_id = mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, "trunk")
                    hits_world = lidar.scan(lidar_origin, Rwb, bodyexclude=trunk_id)
                    
                    # print("Raw hits:", hits_world.shape[0])
                    # if hits_world.shape[0] > 0:
                    #     print("Max hit Z:", hits_world[:,2].max())


                    # 1) drop ground returns (tune threshold)
                    zmin = 0.05              # keep points above 5cm (obstacles)
                    zmax = 1.20              # optional: ignore very high stuff
                    keep_z = (hits_world[:,2] > zmin) & (hits_world[:,2] < zmax)

                    # 2) drop points too close to robot (self/noise)
                    dx = hits_world[:,0] - px
                    dy = hits_world[:,1] - py
                    r = np.sqrt(dx*dx + dy*dy)
                    keep_r = r > 0.45        # keep points farther than ~45cm from COM

                    hits_filt = hits_world[keep_z & keep_r]
                    costmap.update(hits_filt)
                    obstacle_xy = hits_filt[:, :2]
                    visualize_lidar_hits(viewer, hits_filt)

                    # --- UPDATE GLOBAL HEIGHT MAP ---
                    heightmap.update(hits_world)
                    go2.terrain = heightmap

                    # if ctrl_i % 40 == 0:
                    #     debug_plot_heightmap(heightmap.hmap, heightmap.res, heightmap.origin_xy)

                    # downsample for speed (VERY important)
                    if obstacle_xy.shape[0] > 250:
                        idx = np.random.choice(obstacle_xy.shape[0], 250, replace=False)
                        obstacle_xy = obstacle_xy[idx]

                    u0 = mppi.command(state0, goal_xy, obstacle_xy)
                    debug_frames.append({
                        "state": state0.copy(),
                        "U_batch": mppi.last_U_batch.copy(),
                        "costmap": costmap.grid.copy(),
                        "obstacles": obstacle_xy.copy()
                    })


                # if ctrl_i % 10 == 0:  # don’t draw every tick
                #     # pass
                #     debug_plot_mppi(
                #         state0,
                #         goal_xy,
                #         obstacle_xy,
                #         mppi.last_U_batch,
                #         mppi
                #     )


                vx_des_body = float(u0[0])
                vy_des_body = float(u0[1])
                wz_des_body = float(u0[2])
                z_pos_des_body = 0.27
                vx_des_body = np.clip(vx_des_body, -0.8, 0.8)
                vy_des_body = np.clip(vy_des_body, -0.3, 0.3)
                wz_des_body = np.clip(wz_des_body, -0.6, 0.6)
                # vx_des_body = 0.8
                # vy_des_body = 0.0
                # wz_des_body = 0.0
                # z_pos_des_body = 0.27
                traj.generate_traj(
                    go2,
                    gait,
                    time_now_s,
                    vx_des_body,
                    vy_des_body,
                    z_pos_des_body,
                    wz_des_body,
                    time_step=MPC_DT,
                )

                sol = mpc.solve_QP(go2, traj, False)

                mpc_solve_time_ms.append(mpc.solve_time)
                mpc_update_time_ms.append(mpc.update_time)

                N = traj.N
                w_opt = sol["x"].full().flatten()
                X_opt = w_opt[: 12 * (N)].reshape((12, N), order="F")
                U_opt = w_opt[12 * (N) :].reshape((12, N), order="F")

            # Extract first GRF from MPC
            mpc_force_world[:, ctrl_i] = U_opt[:, 0]

            # Compute joint torques
            FL = leg_controller.compute_leg_torque(
                "FL", go2, gait, mpc_force_world[LEG_SLICE["FL"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["FL"], ctrl_i] = FL.tau
            foot_traj.pos_des[LEG_SLICE["FL"], ctrl_i] = FL.pos_des
            foot_traj.pos_now[LEG_SLICE["FL"], ctrl_i] = FL.pos_now
            foot_traj.vel_des[LEG_SLICE["FL"], ctrl_i] = FL.vel_des
            foot_traj.vel_now[LEG_SLICE["FL"], ctrl_i] = FL.vel_now

            FR = leg_controller.compute_leg_torque(
                "FR", go2, gait, mpc_force_world[LEG_SLICE["FR"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["FR"], ctrl_i] = FR.tau
            foot_traj.pos_des[LEG_SLICE["FR"], ctrl_i] = FR.pos_des
            foot_traj.pos_now[LEG_SLICE["FR"], ctrl_i] = FR.pos_now
            foot_traj.vel_des[LEG_SLICE["FR"], ctrl_i] = FR.vel_des
            foot_traj.vel_now[LEG_SLICE["FR"], ctrl_i] = FR.vel_now

            RL = leg_controller.compute_leg_torque(
                "RL", go2, gait, mpc_force_world[LEG_SLICE["RL"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["RL"], ctrl_i] = RL.tau
            foot_traj.pos_des[LEG_SLICE["RL"], ctrl_i] = RL.pos_des
            foot_traj.pos_now[LEG_SLICE["RL"], ctrl_i] = RL.pos_now
            foot_traj.vel_des[LEG_SLICE["RL"], ctrl_i] = RL.vel_des
            foot_traj.vel_now[LEG_SLICE["RL"], ctrl_i] = RL.vel_now

            RR = leg_controller.compute_leg_torque(
                "RR", go2, gait, mpc_force_world[LEG_SLICE["RR"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["RR"], ctrl_i] = RR.tau
            foot_traj.pos_des[LEG_SLICE["RR"], ctrl_i] = RR.pos_des
            foot_traj.pos_now[LEG_SLICE["RR"], ctrl_i] = RR.pos_now
            foot_traj.vel_des[LEG_SLICE["RR"], ctrl_i] = RR.vel_des
            foot_traj.vel_now[LEG_SLICE["RR"], ctrl_i] = RR.vel_now

            # Saturate + hold
            tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
            tau_hold = tau_cmd[:, ctrl_i].copy()

            tau_log_ctrl_Nm[ctrl_i, :] = tau_hold

            ctrl_i += 1

        #Apply held torques at every SIM step
        mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
        mujoco_go2.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
        # viewer.sync()
        #Render-rate logging for smooth replay
        t_after = float(mujoco_go2.data.time)
        if t_after + 1e-12 >= next_render_t:
            time_log_render.append(t_after)
            q_log_render.append(mujoco_go2.data.qpos.copy())
            tau_log_render.append(tau_hold.copy())
            next_render_t += RENDER_DT

sim_end_time = time.perf_counter()
print(
    f"\nSimulation ended."
    f"\nElapsed time: {sim_end_time - sim_start_time:.3f}s"
    f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}"
)

# --------------------------------------------------------------------------------
# Simulation Results
# --------------------------------------------------------------------------------
print("Replaying MPPI debug...")

plt.figure(figsize=(8,8))

for frame in debug_frames:

    plt.clf()

    grid = frame["costmap"]
    res = costmap.res
    origin = costmap.origin_xy

    extent = [
        origin[0],
        origin[0] + grid.shape[1] * res,
        origin[1],
        origin[1] + grid.shape[0] * res,
    ]

    plt.imshow(grid,
               origin="lower",
               extent=extent,
               cmap="hot",
               alpha=0.6,
               vmin=0,
               vmax=1)

    state = frame["state"]
    U_batch = frame["U_batch"]

    X = mppi.rollout(state, U_batch)

    for i in range(min(120, X.shape[0])):
        plt.plot(X[i,:,0], X[i,:,1],
                 color="blue",
                 alpha=0.1)

    plt.scatter(state[0], state[1], c='black', s=60)
    plt.scatter(goal_xy[0], goal_xy[1], c='green', s=100)

    if frame["obstacles"].shape[0] > 0:
        plt.scatter(frame["obstacles"][:,0],
                    frame["obstacles"][:,1],
                    c='cyan', s=5)

    plt.axis("equal")
    plt.xlim(-4,4)
    plt.ylim(-4,4)

    plt.pause(0.03)

plt.show()


# Plot results
t_vec = np.arange(ctrl_i) * CTRL_DT
plot_swing_foot_traj(t_vec, foot_traj, False)
plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)

# Replay simulation
time_log_render = np.asarray(time_log_render, dtype=float)
q_log_render = np.asarray(q_log_render, dtype=float)
tau_log_render = np.asarray(tau_log_render, dtype=float)

mujoco_go2.replay_simulation(time_log_render, q_log_render, tau_log_render, RENDER_DT, REALTIME_FACTOR)
hold_until_all_fig_closed()
