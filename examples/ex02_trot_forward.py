"""
Demo 02: Trot forward
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
import heapq
from dataclasses import dataclass, field

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import Gait
from convex_mpc.plot_helper import plot_mpc_result, plot_swing_foot_traj, plot_solve_time, hold_until_all_fig_closed
import subprocess as _sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Arc
# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

# Simulation Setting
INITIAL_X_POS = -2
INITIAL_Y_POS = 0
# How long does the simulation run for How much time 
RUN_SIM_LENGTH_S = 20.0

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
GAIT_DUTY = 0.8
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

    # Tuning
    ema_alpha_ground: float = 0.25   # smoothing for ground estimate
    ema_alpha_top: float = 0.50      # smoothing for top estimate
    min_points_per_cell: int = 1     # observed mask threshold
    ground_quantile: float = 0.20    # robust "near-ground" percentile (ramp-safe)

    # Fallback height for unobserved cells — set to match the hfield geom z-offset
    # in scene_test_forest.xml: <geom ... pos="0 0 -0.09" .../>.
    # Without this, unobserved cells return z=0 which is 9 cm ABOVE the actual
    # minimum ground surface, causing the swing foot to target the wrong height.
    ground_z_fallback: float = 0.0

    def __post_init__(self):
        self.N = int(self.size_xy / self.res)
        self.origin_xy = np.array([-self.size_xy / 2.0, -self.size_xy / 2.0])

        # Dual layers
        self.h_ground = np.full((self.N, self.N), np.nan, dtype=np.float32)
        self.h_top    = np.full((self.N, self.N), np.nan, dtype=np.float32)

        # Observed count (mask)
        self.count = np.zeros((self.N, self.N), dtype=np.uint16)
        self.obs = np.zeros((self.N, self.N), dtype=np.uint8)

    def world_to_grid(self, x, y):
        ix = ((x - self.origin_xy[0]) / self.res).astype(int)
        iy = ((y - self.origin_xy[1]) / self.res).astype(int)
        return ix, iy

    def update(self, hits_world: np.ndarray):
        
        if hits_world.shape[0] == 0:
            return

        x = hits_world[:, 0]
        y = hits_world[:, 1]
        z = hits_world[:, 2]

        ix, iy = self.world_to_grid(x, y)

        valid = (ix >= 0) & (ix < self.N) & (iy >= 0) & (iy < self.N)
        ix = ix[valid]
        iy = iy[valid]
        z  = z[valid]

        # Group points by cell (simple loop; ok at your rates)
        # For each cell, compute:
        #   ground_z = quantile(z, ground_quantile)
        #   top_z    = max(z)
        # and update with EMA.
        # NOTE: We iterate only over occupied cells for speed.
        lin = iy * self.N + ix
        uniq = np.unique(lin)

        for u in uniq:
            j = u // self.N
            i = u % self.N
            zz = z[lin == u]
            if zz.size == 0:
                continue

            ground_z = float(np.quantile(zz, self.ground_quantile))
            top_z    = float(np.max(zz))

            # EMA update ground
            prev_g = self.h_ground[j, i]
            if np.isnan(prev_g):
                self.h_ground[j, i] = ground_z
            else:
                a = self.ema_alpha_ground
                self.h_ground[j, i] = (1 - a) * prev_g + a * ground_z

            # EMA update top
            prev_t = self.h_top[j, i]
            if np.isnan(prev_t):
                self.h_top[j, i] = top_z
            else:
                a = self.ema_alpha_top
                self.h_top[j, i] = (1 - a) * prev_t + a * top_z

            self.count[j, i] = min(65535, int(self.count[j, i]) + int(zz.size))

    def observed_mask(self):
        return self.count >= self.min_points_per_cell

    def query_ground_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix, iy = self.world_to_grid(x, y)
        valid = (ix >= 0) & (ix < self.N) & (iy >= 0) & (iy < self.N)

        out = np.full(x.shape, self.ground_z_fallback, dtype=float)
        vals = self.h_ground[iy[valid], ix[valid]]
        vals = np.where(np.isnan(vals), self.ground_z_fallback, vals)
        out[valid] = vals
        return out

    def query_top_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix, iy = self.world_to_grid(x, y)
        valid = (ix >= 0) & (ix < self.N) & (iy >= 0) & (iy < self.N)

        # Use ground_z_fallback for unobserved cells so clearance = top - ground = 0,
        # avoiding false obstacle detections in unexplored areas.
        out = np.full(x.shape, self.ground_z_fallback, dtype=float)
        vals = self.h_top[iy[valid], ix[valid]]
        vals = np.where(np.isnan(vals), self.ground_z_fallback, vals)
        out[valid] = vals
        return out

    def query_clearance_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        g = self.query_ground_batch(x, y)
        t = self.query_top_batch(x, y)
        return np.maximum(0.0, t - g)

    def height_and_normal(self, x: float, y: float):

        # Guard against NaNs from upstream
        if not np.isfinite(x) or not np.isfinite(y):
            return self.ground_z_fallback, np.array([0.0, 0.0, 1.0])
        # Use ground layer for normals
        ix = int((x - self.origin_xy[0]) / self.res)
        iy = int((y - self.origin_xy[1]) / self.res)

        if ix < 0 or ix >= self.N or iy < 0 or iy >= self.N:
            return self.ground_z_fallback, np.array([0.0, 0.0, 1.0])

        z = self.h_ground[iy, ix]
        if np.isnan(z):
            return self.ground_z_fallback, np.array([0.0, 0.0, 1.0])

        dzdx = 0.0
        dzdy = 0.0

        if 1 <= ix < self.N-1 and 1 <= iy < self.N-1:
            z_x1 = self.h_ground[iy, ix+1]
            z_x0 = self.h_ground[iy, ix-1]
            z_y1 = self.h_ground[iy+1, ix]
            z_y0 = self.h_ground[iy-1, ix]

            if not np.isnan(z_x1) and not np.isnan(z_x0):
                dzdx = (z_x1 - z_x0) / (2*self.res)
            if not np.isnan(z_y1) and not np.isnan(z_y0):
                dzdy = (z_y1 - z_y0) / (2*self.res)

        normal = np.array([-dzdx, -dzdy, 1.0])
        normal /= (np.linalg.norm(normal) + 1e-9)

        return float(z), normal
    def query_height_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Backward-compatible alias.
        For terrain slope/normals we want the WALKABLE ground surface.
        """
        return self.query_ground_batch(x, y)

    def query_steppability_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Foothold-Quality-Aware (FQA) steppability score.
        Returns a cost in [0, 1] per query point:
          0.0 = perfectly flat, ideal stepping surface
          1.0 = untraversable (steep slope or rough terrain)

        Computed from:
          1. Slope magnitude (terrain gradient)
          2. Surface roughness (height std in a local patch)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        delta = self.res

        # --- Slope component (central differences) ---
        z_xp = self.query_ground_batch(x + delta, y)
        z_xm = self.query_ground_batch(x - delta, y)
        z_yp = self.query_ground_batch(x, y + delta)
        z_ym = self.query_ground_batch(x, y - delta)

        dzdx = (z_xp - z_xm) / (2 * delta)
        dzdy = (z_yp - z_ym) / (2 * delta)
        slope_mag = np.sqrt(dzdx**2 + dzdy**2)

        # Normalize: slope_mag of 0.5 (≈27°) maps to 1.0
        slope_score = np.clip(slope_mag / 0.5, 0.0, 1.0)

        # --- Roughness component (height variance in a 5-point cross) ---
        z_c = self.query_ground_batch(x, y)
        patch = np.stack([z_c, z_xp, z_xm, z_yp, z_ym], axis=0)  # (5, N)
        roughness = np.std(patch, axis=0)

        # Normalize: roughness of 0.03 m maps to 1.0
        rough_score = np.clip(roughness / 0.03, 0.0, 1.0)

        # --- Combined steppability (weighted) ---
        steppability = 0.6 * slope_score + 0.4 * rough_score

        return steppability
    

class ObstacleCostMap2D:
    def __init__(self, size_xy=12.0, res=0.05):
        self.size_xy = size_xy
        self.res = res
        self.N = int(size_xy / res)

        self.origin_xy = np.array([-size_xy/2, -size_xy/2])
        self.grid = np.zeros((self.N, self.N), dtype=np.float32)

        self.decay = 0.98
        self.inflate_radius = 0.50

        # Clearance thresholds (tune)
        self.step_thresh   = 0.06   # rock/log (used later for swing clearance)
        self.lethal_thresh = 0.35   # tree/wall (MPPI must avoid)
    def min_pool_nan(self, a, r):
        """
        r = radius in cells (r=2 means 5x5 neighborhood)
        NaNs ignored. Returns NaN if all neighbors are NaN.
        """
        pad = r
        ap = np.pad(a, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)
        out = np.full_like(a, np.nan, dtype=np.float32)

        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                w = ap[pad+dy:pad+dy+a.shape[0], pad+dx:pad+dx+a.shape[1]]
                # nanmin across shifts
                out = np.fmin(out, w) if np.any(np.isfinite(out)) else w.copy()
                # ^ fmin keeps NaNs weirdly; so do explicit:
        # Better explicit nanmin accumulation:
        out = np.full_like(a, np.inf, dtype=np.float32)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                w = ap[pad+dy:pad+dy+a.shape[0], pad+dx:pad+dx+a.shape[1]]
                out = np.minimum(out, np.where(np.isfinite(w), w, np.inf))
        out = np.where(np.isfinite(out) & (out < np.inf), out, np.nan)
        return out
    def update_from_heightmap(self, heightmap,
                          clearance_lethal=0.25,
                          clearance_soft=0.08,
                          inflate_radius=None):

        if inflate_radius is None:
            inflate_radius = self.inflate_radius

        # --------------------------------------
        # 1️⃣ Compute clearance at heightmap resolution
        # --------------------------------------

        clearance_hm = heightmap.h_top - heightmap.h_ground
        clearance_hm = np.nan_to_num(clearance_hm, nan=0.0)

        # --------------------------------------
        # 2️⃣ Downsample to costmap resolution via MAX pooling
        # --------------------------------------

        scale = int(self.res / heightmap.res)
        if scale < 1:
            scale = 1

        clearance = np.zeros((self.N, self.N), dtype=np.float32)

        for j in range(self.N):
            for i in range(self.N):

                iy0 = j * scale
                iy1 = min((j + 1) * scale, heightmap.N)

                ix0 = i * scale
                ix1 = min((i + 1) * scale, heightmap.N)

                block = clearance_hm[iy0:iy1, ix0:ix1]

                if block.size > 0:
                    clearance[j, i] = np.max(block)

        # print("[DBG] clearance max/mean:",
        #     float(np.max(clearance)),
        #     float(np.mean(clearance)))

        # --------------------------------------
        # 3️⃣ Obstacle classification
        # --------------------------------------

        raw = np.zeros_like(clearance, dtype=np.float32)

        raw[clearance >= clearance_lethal] = 1.0

        # print("Raw obstacle cells:", int(np.sum(raw > 0.05)))
        # print("Raw max:", float(raw.max()))

        # --------------------------------------
        # 4️⃣ Inflation
        # --------------------------------------

        inflated = self.inflate_from_raw(raw)

        self.grid *= self.decay
        self.grid = np.maximum(self.grid, inflated)
    def inflate_from_raw(self, raw):
        inflated = raw.copy()
        radius_cells = int(self.inflate_radius / self.res)
        occ = np.argwhere(raw > 0.05)  # only iterate obstacle cells

        for (j, i) in occ:
            for dx in range(-radius_cells, radius_cells+1):
                for dy in range(-radius_cells, radius_cells+1):
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < self.N and 0 <= nj < self.N:
                        dist = np.sqrt(dx*dx + dy*dy) * self.res
                        seed = raw[j, i]
                        cost = seed * max(0.0, 1.0 - dist/self.inflate_radius)
                        if cost > inflated[nj, ni]:
                            inflated[nj, ni] = cost

        return inflated

    def world_to_grid(self, x, y):
        ix = ((x - self.origin_xy[0]) / self.res).astype(int)
        iy = ((y - self.origin_xy[1]) / self.res).astype(int)
        return ix, iy

    def query_cost_batch(self, x, y):
        ix, iy = self.world_to_grid(x, y)
        valid = (ix>=0)&(ix<self.N)&(iy>=0)&(iy<self.N)
        cost = np.zeros_like(x, dtype=float)
        cost[valid] = self.grid[iy[valid], ix[valid]]
        return cost


# ============================================================================
#  Terrain-Aware A* Global Path Planner
# ============================================================================

class TerrainAwarePlanner:
    """
    Weighted A* planner on the costmap grid.
    Combines obstacle cost (from ObstacleCostMap2D) with terrain
    traversability (slope + steppability from GlobalHeightMap).
    Replans when the current path crosses newly-lethal cells.
    """

    def __init__(self, costmap, heightmap=None):
        self.costmap = costmap
        self.heightmap = heightmap
        self.path = None            # (P, 2) world-frame waypoints resampled at path_spacing
        self.w_obs = 100.0          # obstacle cost weight in A* edges
        self.w_terrain = 5.0        # terrain cost weight in A* edges
        self.lethal_thresh = 0.8    # costmap cells >= this are impassable
        self.path_spacing = 0.15    # resampled path point spacing (m)

    # ------------------------------------------------------------------
    def plan(self, start_xy, goal_xy):
        """Run A* from start_xy to goal_xy (world frame).
        Returns (P, 2) array of world-frame waypoints, or None."""
        grid = self.costmap.grid
        N = self.costmap.N
        res = self.costmap.res
        origin = self.costmap.origin_xy

        sx = int(np.clip((start_xy[0] - origin[0]) / res, 0, N - 1))
        sy = int(np.clip((start_xy[1] - origin[1]) / res, 0, N - 1))
        gx = int(np.clip((goal_xy[0]  - origin[0]) / res, 0, N - 1))
        gy = int(np.clip((goal_xy[1]  - origin[1]) / res, 0, N - 1))

        trav = self._build_cost_grid()

        # Create a mask of the previous path to add "stickiness" / hysteresis
        # so the planner doesn't continuously flip-flop left/right around obstacles.
        prev_path_mask = np.zeros((N, N), dtype=bool)
        if self.path is not None:
            px_idx = np.clip(((self.path[:, 0] - origin[0]) / res).astype(int), 0, N - 1)
            py_idx = np.clip(((self.path[:, 1] - origin[1]) / res).astype(int), 0, N - 1)
            prev_path_mask[py_idx, px_idx] = True

        start = (sx, sy)
        goal  = (gx, gy)

        SQRT2 = 1.4142135
        nbrs = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
                ( 0, -1, 1.0),                  ( 0, 1, 1.0),
                ( 1, -1, SQRT2), ( 1, 0, 1.0),  ( 1, 1, SQRT2)]

        open_heap = [(0.0, 0.0, start)]
        came_from = {}
        g_score = np.full((N, N), np.inf, dtype=np.float32)
        g_score[sy, sx] = 0.0
        closed = np.zeros((N, N), dtype=bool)

        while open_heap:
            f, g, (cx, cy) = heapq.heappop(open_heap)

            if (cx, cy) == goal:
                path = [(cx, cy)]
                node = (cx, cy)
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                path_world = np.array([
                    [origin[0] + (ix + 0.5) * res,
                     origin[1] + (iy + 0.5) * res]
                    for (ix, iy) in path
                ])
                self.path = self._smooth(self._resample(path_world))
                print(f"[Planner] Path found: {len(self.path)} pts, "
                      f"{self._arc_length(self.path):.2f}m")
                return self.path

            if closed[cy, cx]:
                continue
            closed[cy, cx] = True

            for dx, dy, step in nbrs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N and not closed[ny, nx]:
                    c = trav[ny, nx]
                    if c >= 1e6:
                        continue
                        
                    # Base step cost
                    step_cost = step * res * (1.0 + c)
                    
                    # Hysteresis: discount cells that were on the previous path
                    if prev_path_mask[ny, nx]:
                        step_cost *= 0.5  # 50% discount encourages sticking to the old plan

                    ng = g + step_cost
                    if ng < g_score[ny, nx]:
                        g_score[ny, nx] = ng
                        h = np.sqrt((nx - gx)**2 + (ny - gy)**2) * res
                        # Weighted A* (weight=1.5) for much faster planning
                        heapq.heappush(open_heap, (ng + 1.5 * h, ng, (nx, ny)))
                        came_from[(nx, ny)] = (cx, cy)

        print("[Planner] No path found!")
        return None

    # ------------------------------------------------------------------
    def needs_replan(self):
        """True if current path crosses newly-lethal cells."""
        if self.path is None:
            return True
        costs = self.costmap.query_cost_batch(self.path[:, 0], self.path[:, 1])
        return bool(np.any(costs >= self.lethal_thresh))

    # ------------------------------------------------------------------
    def _build_cost_grid(self):
        """Combined obstacle + terrain traversability cost grid."""
        grid = self.costmap.grid
        N = self.costmap.N
        
        # Make the inflation curve exponentially steep so A* strongly prefers 
        # staying out of the inflation bubble rather than just taking a shorter 
        # path through it.
        obs_penalty = self.w_obs * (np.exp(grid * 4.0) - 1.0)
        
        trav = np.where(grid >= self.lethal_thresh, 1e6,
                        obs_penalty).astype(np.float32)

        if self.heightmap is not None:
            res = self.costmap.res
            origin = self.costmap.origin_xy
            cx = np.arange(N) * res + origin[0] + res / 2
            cy = np.arange(N) * res + origin[1] + res / 2
            xx, yy = np.meshgrid(cx, cy)
            step = self.heightmap.query_steppability_batch(
                xx.reshape(-1), yy.reshape(-1)).reshape(N, N)
            trav += self.w_terrain * step

        return trav

    # ------------------------------------------------------------------
    def _resample(self, path):
        """Resample path to uniform spacing."""
        if len(path) < 2:
            return path
        diffs = np.diff(path, axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_len)])
        total = cum_len[-1]
        if total < self.path_spacing:
            return path
        n_pts = max(2, int(total / self.path_spacing) + 1)
        s_new = np.linspace(0, total, n_pts)
        x_new = np.interp(s_new, cum_len, path[:, 0])
        y_new = np.interp(s_new, cum_len, path[:, 1])
        return np.column_stack([x_new, y_new])

    @staticmethod
    def _arc_length(path):
        if len(path) < 2:
            return 0.0
        d = np.diff(path, axis=0)
        return float(np.sqrt((d ** 2).sum(axis=1)).sum())

    # ------------------------------------------------------------------
    def _check_segment_clear(self, p0, p1, n_samples=12):
        """True if the line from p0 to p1 stays below lethal_thresh."""
        xs = np.linspace(p0[0], p1[0], n_samples)
        ys = np.linspace(p0[1], p1[1], n_samples)
        costs = self.costmap.query_cost_batch(xs, ys)
        return bool(np.all(costs < self.lethal_thresh))

    def _smooth(self, path, n_lap=40, alpha=0.25):
        """Smooth an A* path in one stage:
        
        Laplacian smoothing — iteratively average interior points with
        their neighbours, keeping start/end pinned. Rounds strict A* grid 
        movements into smooth gentle arcs without destroying complex curves 
        like a greedy line-of-sight shortcutter does.
        """
        if len(path) < 3:
            return path

        # Resample to uniform spacing first so Laplacian smoothing works evenly
        path = self._resample(path)

        if len(path) < 3:
            return path

        if len(path) < 3:
            return path

        # --- Stage 2: Laplacian smoothing (pin start and end) ---
        smoothed = path.copy()
        for _ in range(n_lap):
            prev = smoothed.copy()
            for k in range(1, len(smoothed) - 1):
                candidate = (1 - alpha) * prev[k] + alpha * 0.5 * (prev[k-1] + prev[k+1])
                # Only accept the smoothed point if it stays in free space
                if self.costmap.query_cost_batch(
                        np.array([candidate[0]]),
                        np.array([candidate[1]]))[0] < self.lethal_thresh:
                    smoothed[k] = candidate
        return smoothed


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
# Go2 hip offsets in body frame (x_forward, y_left) — for FQA steppability
_HIP_OFFSETS_BODY = np.array([
    [ 0.19,  0.05],   # FL
    [ 0.19, -0.05],   # FR
    [-0.19,  0.05],   # RL
    [-0.19, -0.05],   # RR
], dtype=float)

class Nav2StyleMPPI:

    def __init__(self, dt):

        # --- MPPI parameters ---
        self.dt = dt
        self.H = 80
        self.BATCH = 400
        self.ITERS = 5

        self.LAMBDA = 8.0
        self.ALPHA = 0.5         # moderate correlation for path following

        self.vx_min, self.vx_max = -0.25, 0.75
        self.vy_min, self.vy_max = -0.5, 0.5
        self.wz_min, self.wz_max = -1.75, 1.75

        self.costmap = None
        self.terrain = None

        self.best_traj = np.zeros((self.H, 3))
        self.std = np.array([0.30, 0.50, 0.50])

        # persistent control sequence
        self.U = np.zeros((self.H, 3))
        self._near_obstacle = False

        # Stuck detection
        self._dist_history = []
        self._stuck_counter = 0

        # Acceleration rate limits
        self._prev_u0 = np.zeros(3)
        self._accel_max = np.array([2.0, 1.5, 10.0])
        self._mppi_dt = dt * 2

        # --- Path-following state ---
        self.path_xy = None       # (P, 2) global path waypoints
        self.path_tangent = None  # (P, 2) unit tangent at each waypoint
        self.path_cumlen = None   # (P,)   cumulative arc length

    def set_costmap(self, costmap):
        self.costmap = costmap
    def set_terrain(self, heightmap):
        self.terrain = heightmap

    def set_path(self, path_xy):
        """Set global path for path-following critics.

        Also warm-starts U along the initial path tangent so rollouts are
        immediately centred around forward motion instead of zero.
        Without this seed, the zero-mean noise on U=0 produces a stationary
        cloud of rollouts with no net progress gradient.
        """
        if path_xy is None or len(path_xy) < 2:
            self.path_xy = None
            self.path_tangent = None
            self.path_cumlen = None
            return
        self.path_xy = path_xy.copy()
        diffs = np.diff(path_xy, axis=0)
        lengths = np.sqrt((diffs ** 2).sum(axis=1))
        tangent = diffs / np.maximum(lengths[:, None], 1e-6)
        self.path_tangent = np.vstack([tangent, tangent[-1:]])
        self.path_cumlen = np.concatenate([[0], np.cumsum(lengths)])

        # Warm-start U: seed each horizon step with a body-frame velocity
        # pointing along the path tangent at that lookahead arc-length.
        # This gives the softmax a meaningful gradient from the very first
        # iteration instead of needing many iterations to escape U=0.
        CRUISE_VX = 0.40   # m/s body-frame forward seed (conservative)
        cumlen = self.path_cumlen
        total_len = cumlen[-1] if cumlen[-1] > 0.01 else 1.0
        for t in range(self.H):
            s = min(t * self.dt * CRUISE_VX, total_len)
            # Index of path point closest to lookahead distance s
            idx = int(np.searchsorted(cumlen, s, side='right')) - 1
            idx = np.clip(idx, 0, len(self.path_tangent) - 1)
            tang = self.path_tangent[idx]   # (2,) unit vector in world frame
            # Seed as pure forward body velocity. MPPI will randomly sample wz
            # to discover that turning is required to follow the path.
            # Previously, we mapped world-frame tangent to body-frame vy, 
            # causing the robot to sidestep (crab-walk) instead of turn.
            self.U[t, 0] = CRUISE_VX
            self.U[t, 1] = 0.0
            self.U[t, 2] = 0.0


    # --------------------------------------
    # correlated noise
    # --------------------------------------
    def correlated_noise(self):
        """
        Pure Gaussian perturbations centered on zero, applied on top of self.U.
        Temporally correlated via an AR(1) filter (ALPHA controls smoothness).
        Distribution is intentionally wide to ensure broad exploration.
        """
        eps = np.random.randn(self.BATCH, self.H, 3)
        eps *= self.std

        # AR(1) temporal smoothing: makes noise correlated across time steps
        # (produces smooth arcs rather than jerky random paths)
        for t in range(1, self.H):
            eps[:, t, :] = (
                self.ALPHA * eps[:, t-1, :]
                + (1.0 - self.ALPHA) * eps[:, t, :]
            )

        return eps

    # --------------------------------------
    # rollout model
    # --------------------------------------
    def rollout(self, state, U_batch):

        B, T, _ = U_batch.shape
        X = np.zeros((B, T, 6))

        x = np.full(B, state[0])
        y = np.full(B, state[1])
        yaw = np.full(B, state[2])
        vx = np.full(B, state[3])
        vy = np.full(B, state[4])
        wz = np.full(B, state[5])

        tau = 0.4  # time constant (tune)

        for t in range(T):

            vx_cmd = np.clip(U_batch[:, t, 0], self.vx_min, self.vx_max)
            vy_cmd = np.clip(U_batch[:, t, 1], self.vy_min, self.vy_max)
            wz_cmd = np.clip(U_batch[:, t, 2], self.wz_min, self.wz_max)

            # first-order velocity tracking
            vx += (vx_cmd - vx) * self.dt / tau
            vy += (vy_cmd - vy) * self.dt / tau
            wz += (wz_cmd - wz) * self.dt / tau

            x += (vx*np.cos(yaw) - vy*np.sin(yaw)) * self.dt
            y += (vx*np.sin(yaw) + vy*np.cos(yaw)) * self.dt
            yaw += wz * self.dt

            X[:, t, 0] = x
            X[:, t, 1] = y
            X[:, t, 2] = yaw
            X[:, t, 3] = vx
            X[:, t, 4] = vy
            X[:, t, 5] = wz

        return X

    # =============================================
    #  Nav2-style MPPI critics
    # =============================================

    def cost(self, X, U_batch, obstacle_xy):
        """Path-following MPPI cost (Nav2-style critics)."""

        x   = X[:, :, 0]
        y   = X[:, :, 1]
        yaw = X[:, :, 2]
        vx  = X[:, :, 3]
        vy  = X[:, :, 4]
        wz  = X[:, :, 5]

        B, H = x.shape
        total = np.zeros(B)

        # Subsample timesteps for vectorised path queries
        step = max(1, H // 30)
        t_idx = np.arange(0, H, step)
        Hs = len(t_idx)

        path_cost = np.zeros(B)
        progress  = np.zeros(B)

        # =============================================
        #  PATH-FOLLOWING CRITICS
        # =============================================
        if self.path_xy is not None and len(self.path_xy) >= 2:
            path = self.path_xy          # (P, 2)
            tangent = self.path_tangent  # (P, 2)
            cumlen = self.path_cumlen    # (P,)
            P = len(path)

            # Trim path: only keep waypoints ahead of the robot
            robot_xy = np.array([x[0, 0], y[0, 0]])
            d_robot = np.sum((path - robot_xy) ** 2, axis=1)
            start_idx = max(0, int(np.argmin(d_robot)) - 1)
            path    = path[start_idx:]
            tangent = tangent[start_idx:]
            cumlen  = cumlen[start_idx:] - cumlen[start_idx]
            P = len(path)

            tx   = x[:, t_idx]      # (B, Hs)
            ty   = y[:, t_idx]
            tyaw = yaw[:, t_idx]

            # Distance from every (batch, timestep) to every path point
            dx = tx[:, :, None] - path[None, None, :, 0]   # (B, Hs, P)
            dy = ty[:, :, None] - path[None, None, :, 1]
            d2 = dx ** 2 + dy ** 2                          # (B, Hs, P)

            closest_idx = np.argmin(d2, axis=2)             # (B, Hs)
            min_dist    = np.sqrt(np.min(d2, axis=2))       # (B, Hs)

            # --- 1. PathFollowCritic — cross-track error ---
            path_cost = min_dist.mean(axis=1)
            total += 80.0 * path_cost

            # --- 2. PathAngleCritic — heading aligned with path tangent ---
            ci = closest_idx.reshape(-1)
            tang_heading = np.arctan2(
                tangent[ci, 1], tangent[ci, 0]).reshape(B, Hs)
            heading_err = np.abs(np.arctan2(
                np.sin(tyaw - tang_heading),
                np.cos(tyaw - tang_heading)))
            total += 120.0 * heading_err.mean(axis=1)

            # --- 2b. Turn-In-Place Critic — restrict forward speed if misaligned ---
            # If the heading error is large, heavily penalize forward velocity (vx).
            # This teaches MPPI to slow down or stop `vx` immediately so it can
            # execute sharp turns (`wz`) to align with the path before proceeding.
            tvx = vx[:, t_idx]
            # Only penalize if heading error is above a small threshold (e.g. 0.2 rad)
            turn_penalty = (np.clip(tvx, 0.0, None) ** 2) * (np.clip(heading_err - 0.2, 0.0, None) ** 2)
            total += 500.0 * turn_penalty.mean(axis=1)

            # --- 3. PathProgressCritic — reward reaching further along path ---
            # Use MAX progress over all subsampled timesteps (not just terminal).
            # This gives a proper gradient even when rollouts barely move forward:
            # a rollout that probes further along the path at any point is rewarded.
            progress_per_t = cumlen[np.clip(closest_idx, 0, len(cumlen) - 1)]  # (B, Hs)
            progress = progress_per_t.max(axis=1)  # (B,) — furthest reach along path
            total_len = cumlen[-1] if cumlen[-1] > 0.01 else 1.0
            total += 20.0 * (total_len - progress) / total_len

            # --- 4. Velocity damping near end of path ---
            end_xy = self.path_xy[-1]
            dist_to_end = np.sqrt(
                (x[:, 0] - end_xy[0]) ** 2 + (y[:, 0] - end_xy[1]) ** 2)
            near_end = (dist_to_end < 1.0).astype(float)
            speed_T = np.sqrt(X[:, -1, 3] ** 2 + X[:, -1, 4] ** 2)
            total += 20.0 * near_end * speed_T

        # =============================================
        #  OBSTACLE CRITIC (graduated per-step)
        # =============================================
        obs_critic = np.zeros(B)
        if self.costmap is not None:
            cost_vals = self.costmap.query_cost_batch(
                x.reshape(-1), y.reshape(-1)).reshape(B, H)
            step_penalty = np.where(
                cost_vals >= 1.0, 300.0,
                np.where(cost_vals > 0.3, 150.0 * cost_vals,
                         30.0 * cost_vals))
            obs_critic = (step_penalty.sum(axis=1) / 20.0
                          + step_penalty.max(axis=1) * 3.0)
            total += obs_critic

        # =============================================
        #  CONSTRAINT CRITIC
        # =============================================
        vx_cmd = U_batch[:, :, 0]
        vy_cmd = U_batch[:, :, 1]
        wz_cmd = U_batch[:, :, 2]
        vx_over = (np.maximum(vx_cmd - self.vx_max, 0.0)
                   + np.maximum(self.vx_min - vx_cmd, 0.0))
        vy_over = (np.maximum(vy_cmd - self.vy_max, 0.0)
                   + np.maximum(self.vy_min - vy_cmd, 0.0))
        wz_over = (np.maximum(wz_cmd - self.wz_max, 0.0)
                   + np.maximum(self.wz_min - wz_cmd, 0.0))
        total += 4.0 * (vx_over + vy_over + wz_over).mean(axis=1)

        # =============================================
        #  SLOPE CRITIC
        # =============================================
        if self.terrain is not None:
            delta = self.terrain.res
            z_x1 = self.terrain.query_height_batch(
                (x + delta).reshape(-1), y.reshape(-1)).reshape(B, H)
            z_x2 = self.terrain.query_height_batch(
                (x - delta).reshape(-1), y.reshape(-1)).reshape(B, H)
            z_y1 = self.terrain.query_height_batch(
                x.reshape(-1), (y + delta).reshape(-1)).reshape(B, H)
            z_y2 = self.terrain.query_height_batch(
                x.reshape(-1), (y - delta).reshape(-1)).reshape(B, H)
            dzdx = (z_x1 - z_x2) / (2 * delta)
            dzdy = (z_y1 - z_y2) / (2 * delta)
            total += 2.5 * np.sqrt(dzdx ** 2 + dzdy ** 2).mean(axis=1)

        # =============================================
        #  STEPPABILITY CRITIC
        # =============================================
        total += self._steppability_cost(x, y, yaw)

        # =============================================
        #  SMOOTHNESS CRITIC
        # =============================================
        dU = np.diff(U_batch, axis=1)
        total += 0.1 * np.abs(dU).mean(axis=(1, 2))

        # --- logging (best trajectory) ---
        bi = int(np.argmin(total))
        self._last_critics = {
            'path':     float(path_cost[bi]),
            'obs':      float(obs_critic[bi]),
            'progress': float(progress[bi]),
        }

        return total





    # --------------------------------------
    # FQA: Foothold-Quality-Aware cost
    # --------------------------------------
    def _steppability_cost(self, x, y, yaw):
        """
        Foothold-Quality-Aware (FQA) MPPI cost term.

        For each rollout position, estimates footholds at the 4 hip
        locations rotated by the rollout heading, queries terrain
        steppability, and penalizes trajectories where feet would
        land on poor surfaces (steep, rough, or uneven ground).

        Args:
            x:   (B, H) rollout x-positions
            y:   (B, H) rollout y-positions
            yaw: (B, H) rollout yaw angles
        Returns:
            cost: (B,) steppability penalty per trajectory
        """
        if self.terrain is None:
            return np.zeros(x.shape[0])

        B, H = x.shape
        cos_yaw = np.cos(yaw)  # (B, H)
        sin_yaw = np.sin(yaw)  # (B, H)

        total_step_cost = np.zeros(B)

        for hx, hy in _HIP_OFFSETS_BODY:
            # Predicted foot position in world frame
            fx = x + hx * cos_yaw - hy * sin_yaw  # (B, H)
            fy = y + hx * sin_yaw + hy * cos_yaw  # (B, H)

            # Query steppability at all foot positions
            scores = self.terrain.query_steppability_batch(
                fx.reshape(-1), fy.reshape(-1)
            ).reshape(B, H)

            # Accumulate per-trajectory mean steppability cost
            total_step_cost += (scores ** 2).mean(axis=1)

        # Average over 4 feet, apply weight
        w_step = 3.0  # tunable weight
        return w_step * (total_step_cost / 4.0)

    # --------------------------------------
    # main step
    # --------------------------------------
    def command(self, state, obstacle_xy):
        """Path-following MPPI command. Uses self.path_xy set via set_path()."""

        # --- Per-call setup (run once, not inside ITERS loop) ---
        if self.path_xy is not None and len(self.path_xy) >= 2:
            end = self.path_xy[-1]
            dist = np.hypot(state[0] - end[0], state[1] - end[1])
        else:
            dist = float('inf')

        if dist < 0.2:
            self.U[:] = 0.0
            self.best_traj[:] = 0.0
            return np.array([0.0, 0.0, 0.0])

        # --- Adaptive noise std (once per call) ---
        critics = getattr(self, '_last_critics', {})
        c_v = 0.0
        if self.costmap is not None:
            c_v = float(self.costmap.query_cost_batch(
                np.array([state[0]]), np.array([state[1]]))[0])
        self._near_obstacle = (c_v > 0.01) or critics.get('obs', 0) > 5.0

        self._dist_history.append(dist)
        if len(self._dist_history) > 10:
            self._dist_history.pop(0)
        if len(self._dist_history) >= 6:
            prog = self._dist_history[0] - self._dist_history[-1]
            if prog < 0.1:
                self._stuck_counter = min(self._stuck_counter + 1, 30)
            else:
                self._stuck_counter = max(0, self._stuck_counter - 2)
        stuck = self._stuck_counter > 3

        if dist < 1.5 and critics.get('path', 1.0) < 0.3:
            self.std = np.array([0.08, 0.05, 0.40])
        elif stuck or self._near_obstacle:
            self.std = np.array([0.30, 0.10, 1.20])
        else:
            self.std = np.array([0.35, 0.05, 1.00])

        # --- Re-seed U from path tangent + current yaw if U is weak ---
        # Prevents U from staying near zero after replanning or after the
        # receding horizon drains the warm-start.
        u_fwd_mean = float(self.U[:, 0].mean())
        if self.path_xy is not None and abs(u_fwd_mean) < 0.10:
            robot_xy = np.array([state[0], state[1]])
            robot_yaw = state[2]
            d_robot = np.sum((self.path_xy - robot_xy) ** 2, axis=1)
            start_idx = int(np.argmin(d_robot))
            for t in range(self.H):
                # Lookahead distance
                s = t * self.dt * 0.35 
                idx = int(np.searchsorted(self.path_cumlen, s + self.path_cumlen[start_idx], side='right')) - 1
                idx = np.clip(idx, 0, len(self.path_tangent) - 1)
                tang = self.path_tangent[idx]  # world-frame unit tangent
                
                # Desired heading from tangent
                tang_heading = np.arctan2(tang[1], tang[0])
                heading_err = np.arctan2(np.sin(tang_heading - robot_yaw), 
                                         np.cos(tang_heading - robot_yaw))
                                         
                # Smart Initial Guess: limit vx if heading is wrong, and seed wz proportionally
                seed_vx = 0.35 if abs(heading_err) < 0.3 else 0.0
                seed_wz = 2.5 * heading_err
                
                self.U[t, 0] = seed_vx
                self.U[t, 1] = 0.0    # Never crab
                self.U[t, 2] = seed_wz

        # --- MPPI iterations ---
        for it in range(self.ITERS):
            eps = self.correlated_noise()
            U_batch = self.U[None, :, :] + eps

            X = self.rollout(state, U_batch)
            costs = self.cost(X, U_batch, obstacle_xy)

            if not np.all(np.isfinite(costs)):
                costs = np.where(np.isfinite(costs), costs, 1e6)

            best_cost = np.min(costs)

            # --- Softmax update ---
            # Use self.LAMBDA as temperature (was hardcoded 0.3 which collapsed
            # weights to near-argmin, destroying proper rollout blending).
            beta = costs.min()
            w = np.exp(-(costs - beta) / self.LAMBDA)
            w_sum = w.sum()
            if w_sum <= 1e-9 or not np.isfinite(w_sum):
                w = np.ones_like(w) / len(w)
            else:
                w /= w_sum
            self.U = np.sum(w[:, None, None] * U_batch, axis=0)
            self.U[:, 0] = np.clip(self.U[:, 0], self.vx_min, self.vx_max)
            self.U[:, 1] = np.clip(self.U[:, 1], self.vy_min, self.vy_max)
            self.U[:, 2] = np.clip(self.U[:, 2], self.wz_min, self.wz_max)

        critics = getattr(self, '_last_critics', {})
        print(f"[MPPI] dist={dist:.2f} best={best_cost:.1f} "
              f"path={critics.get('path',0):.2f} "
              f"obs={critics.get('obs',0):.1f} "
              f"prog={critics.get('progress',0):.2f} "
              f"stk={self._stuck_counter}")

        # Execute first control
        u0 = self.U[0].copy()

        # Acceleration clamping
        du_max = self._accel_max * self._mppi_dt
        du = np.clip(u0 - self._prev_u0, -du_max, du_max)
        u0 = self._prev_u0 + du
        self._prev_u0 = u0.copy()

        self.last_U_plan = self.U.copy()

        # Receding horizon
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]
        self.last_U_batch = U_batch

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

        for i in range(min(1200, X.shape[0])):
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


if __name__ == "__main__":
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
    # ground_z_fallback matches the hfield geom z-offset in scene_test_forest.xml
    # (<geom ... pos="0 0 -0.09" hfield="forest" .../>), so that unobserved cells
    # return the correct baseline ground height instead of world-z 0.
    heightmap = GlobalHeightMap(size_xy=12.0, res=0.05, ground_z_fallback=-0.09)
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

    # --- Global path planner ---
    planner = TerrainAwarePlanner(costmap, heightmap)
    initial_path = planner.plan(np.array([INITIAL_X_POS, INITIAL_Y_POS]), goal_xy)
    if initial_path is not None:
        mppi.set_path(initial_path)

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
                    vx_world = x_vec[6, ctrl_i]
                    vy_world = x_vec[7, ctrl_i]
                    wz_world = x_vec[11, ctrl_i]

                    Rwb = go2.R_world_to_body  # 3x3
                    v_body = Rwb @ np.array([vx_world, vy_world, 0.0])
                    vx_body, vy_body = v_body[0], v_body[1]
                    state0 = np.array([px, py, yaw, vx_body, vy_body, wz_world])


                    if (ctrl_i % (4 * STEPS_PER_MPC)) == 0:
                        # LiDAR origin (a bit above COM so it doesn't hit the ground instantly)
                        lidar_origin = go2.current_config.base_pos.copy()
                        lidar_origin[2] -= 0.1

                        base_body_id = mj.mj_name2id(
                            mujoco_go2.model,
                            mj.mjtObj.mjOBJ_BODY,
                            "trunk"
                        )

                        Rwb = mujoco_go2.data.xmat[base_body_id].reshape(3,3).copy()

                        # Perform scan using full base rotation
                        trunk_id = mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, "trunk")
                        hits_world = lidar.scan(lidar_origin, Rwb, bodyexclude=trunk_id)

                        # Keep BOTH ground + obstacle points. Only remove robot-near / extreme outliers.
                        zmin = -0.50
                        zmax =  2.00
                        keep_z = (hits_world[:, 2] > zmin) & (hits_world[:, 2] < zmax)

                        dx = hits_world[:, 0] - px
                        dy = hits_world[:, 1] - py
                        r = np.sqrt(dx*dx + dy*dy)
                        keep_r = r > 0.45

                        hits_filt = hits_world[keep_z & keep_r]

                        # Update maps
                        heightmap.update(hits_filt)
                        go2.terrain = heightmap
                        costmap.update_from_heightmap(heightmap, clearance_lethal=0.12, clearance_soft=0.05)

                        # --- Replan global path periodically from current position ---
                        new_path = planner.plan(np.array([px, py]), goal_xy)
                        if new_path is not None:
                            mppi.set_path(new_path)
                            # print(f"[Planner] Replanned: {len(new_path)} pts")

                        # Compute clearance per hit cell (object vs ground)
                        ix, iy = heightmap.world_to_grid(hits_filt[:,0], hits_filt[:,1])
                        valid = (ix>=0)&(ix<heightmap.N)&(iy>=0)&(iy<heightmap.N)

                        clear = np.zeros(len(hits_filt))
                        clear[valid] = (
                            heightmap.h_top[iy[valid], ix[valid]] -
                            heightmap.h_ground[iy[valid], ix[valid]]
                        )

                        obstacle_mask = clear > 0.08
                        obstacle_xy = hits_filt[obstacle_mask, :2]

                        if obstacle_xy.shape[0] > 250:
                            idx = np.random.choice(obstacle_xy.shape[0], 250, replace=False)
                            obstacle_xy = obstacle_xy[idx]

                    # --- MPPI runs at 2× MPC rate ---
                    if (ctrl_i % (2 * STEPS_PER_MPC)) == 0:
                        u0 = mppi.command(state0, obstacle_xy)

                    # Log debug frame (includes path for visualization)
                    if (ctrl_i % (2 * STEPS_PER_MPC)) == 0:
                        debug_frames.append({
                            "state": state0.copy(),
                            "u0": u0.copy(),
                            "U_batch": mppi.last_U_batch.copy(),
                            "U_plan":  mppi.last_U_plan.copy(),
                            "costmap": costmap.grid.copy(),
                            "obstacles": obstacle_xy.copy(),
                            "path": mppi.path_xy.copy() if mppi.path_xy is not None else None,
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
                    vy_des_body = np.clip(vy_des_body, -0.5, 0.5)
                    wz_des_body = np.clip(wz_des_body, -1.5, 1.5)
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
                    # if ctrl_i % 20 == 0:
                    #     print("norm_z min/mean:", traj.contact_normals[:,:,2].min(), traj.contact_normals[:,:,2].mean())

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
    # blocker = input("Press Enter to continue...")

    print("Rendering MPPI debug video...")

    from matplotlib.lines import Line2D

    # Explicitly point matplotlib at the conda-env ffmpeg so it works under sudo
    # (sudo strips PATH, so the system can't find the conda-installed ffmpeg).
    _FFMPEG_PATH = "/home/suleiman/miniconda3/envs/go2-convex-mpc/bin/ffmpeg"
    if not matplotlib.rcParams.get('animation.ffmpeg_path') or \
            matplotlib.rcParams['animation.ffmpeg_path'] == 'ffmpeg':
        matplotlib.rcParams['animation.ffmpeg_path'] = _FFMPEG_PATH

    base_name = "mppi_debug"
    ext = ".mp4"
    MPPI_VIDEO_PATH = os.path.abspath(f"{base_name}{ext}")
    counter = 1
    while os.path.exists(MPPI_VIDEO_PATH):
        MPPI_VIDEO_PATH = os.path.abspath(f"{base_name}_{counter}{ext}")
        counter += 1
    VIDEO_FPS = 25

    fig, ax = plt.subplots(figsize=(8, 8))
    writer = FFMpegWriter(fps=VIDEO_FPS, metadata={"title": "MPPI Debug"}, bitrate=3000)

    ARROW_SCALE = 0.8
    YAW_ARC_RADIUS = 0.25

    with writer.saving(fig, MPPI_VIDEO_PATH, dpi=120):
        for fi, frame in enumerate(debug_frames):

            ax.cla()

            grid = frame["costmap"]
            res = costmap.res
            origin = costmap.origin_xy
            extent = [
                origin[0],
                origin[0] + grid.shape[1] * res,
                origin[1],
                origin[1] + grid.shape[0] * res,
            ]
            ax.imshow(grid, origin="lower", extent=extent, cmap="hot", alpha=0.6,
                      vmin=0, vmax=max(1e-3, float(grid.max())))

            state   = frame["state"]
            u0      = frame["u0"]
            U_batch = frame["U_batch"]

            px, py, yaw = state[0], state[1], state[2]
            vx_body, vy_body, wz_actual = state[3], state[4], state[5]
            vx_cmd, vy_cmd, wz_cmd = u0[0], u0[1], u0[2]

            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

            actual_wx  = vx_body * cos_yaw - vy_body * sin_yaw
            actual_wy  = vx_body * sin_yaw + vy_body * cos_yaw
            desired_wx = vx_cmd * cos_yaw - vy_cmd * sin_yaw
            desired_wy = vx_cmd * sin_yaw + vy_cmd * cos_yaw

            # Global path
            path = frame.get("path")
            if path is not None:
                ax.plot(path[:, 0], path[:, 1], color='white', linewidth=3.0, zorder=3, alpha=0.9)
                ax.plot(path[:, 0], path[:, 1], color='cyan',  linewidth=1.5, zorder=4,
                        linestyle='--', label='Global path')

            # Rollout cloud
            X = mppi.rollout(state, U_batch)
            for i in range(min(120, X.shape[0])):
                ax.plot(X[i, :, 0], X[i, :, 1], color="blue", alpha=0.08)

            # Best plan trajectory
            U_plan = frame.get("U_plan")
            if U_plan is not None:
                X_plan = mppi.rollout(state, U_plan[None, :, :])
                ax.plot(X_plan[0, :, 0], X_plan[0, :, 1],
                        color="yellow", linewidth=2.5, zorder=6, label="MPPI plan")
                wz_plan = float(U_plan[0, 2])
            else:
                wz_plan = float('nan')

            # Robot + goal
            ax.scatter(px, py, c='black', s=80, zorder=5)
            ax.scatter(goal_xy[0], goal_xy[1], c='limegreen', s=120, zorder=5,
                       edgecolors='darkgreen', linewidths=1.5)

            # Heading line
            head_len = 0.20
            ax.plot([px, px + head_len * cos_yaw], [py, py + head_len * sin_yaw],
                    color='black', linewidth=2.5, solid_capstyle='round', zorder=6)

            # Velocity arrows
            des_speed = np.sqrt(desired_wx**2 + desired_wy**2)
            if des_speed > 0.01:
                ax.annotate('', xy=(px + desired_wx * ARROW_SCALE, py + desired_wy * ARROW_SCALE),
                            xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color='limegreen', lw=2.5, mutation_scale=15),
                            zorder=7)
            act_speed = np.sqrt(actual_wx**2 + actual_wy**2)
            if act_speed > 0.01:
                ax.annotate('', xy=(px + actual_wx * ARROW_SCALE, py + actual_wy * ARROW_SCALE),
                            xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, mutation_scale=15),
                            zorder=7)

            # Yaw-rate arcs
            yaw_deg = np.degrees(yaw)
            for wz_val, color, ls in [(wz_cmd, 'limegreen', '--'), (wz_actual, 'red', '-')]:
                if abs(wz_val) > 0.02:
                    sweep = np.clip(np.degrees(wz_val) * 0.5, -90, 90)
                    arc = Arc((px, py), 2*YAW_ARC_RADIUS, 2*YAW_ARC_RADIUS,
                              angle=yaw_deg,
                              theta1=0 if sweep > 0 else sweep,
                              theta2=sweep if sweep > 0 else 0,
                              color=color, lw=2.5, linestyle=ls, zorder=7)
                    ax.add_patch(arc)

            # LiDAR points
            if frame["obstacles"].shape[0] > 0:
                ax.scatter(frame["obstacles"][:, 0], frame["obstacles"][:, 1], c='cyan', s=5)

            # Legend
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], color='limegreen', lw=2.5, label=f'MPPI desired (v={des_speed:.2f} m/s)'),
                Line2D([0],[0], color='red',       lw=2.5, label=f'Actual (v={act_speed:.2f} m/s)'),
                Line2D([0],[0], color='black',     lw=2.5, label=f'Heading (yaw={np.degrees(yaw):.1f}°)'),
                Line2D([0],[0], color='limegreen', lw=2, linestyle='--', label=f'Des ωz={wz_cmd:.2f} rad/s'),
                Line2D([0],[0], color='red',       lw=2,                label=f'Act ωz={wz_actual:.2f} rad/s'),
            ], loc='upper right', fontsize=8)

            ax.set_title(
                f'Frame {fi+1}/{len(debug_frames)}  |  '
                f'plan ωz={wz_plan:.2f}  →  cmd ωz={wz_cmd:.2f}  (act ωz={wz_actual:.2f})',
                fontsize=9)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

            writer.grab_frame()
            if (fi + 1) % 10 == 0:
                print(f"  Rendered {fi+1}/{len(debug_frames)} frames...", flush=True)

    plt.close(fig)
    print(f"\\n{'='*60}")
    print(f"✅ MPPI Debug Video saved successfully!")
    print(f"Location: {MPPI_VIDEO_PATH}")
    print(f"You can open it manually with: vlc {MPPI_VIDEO_PATH}")
    print(f"{'='*60}\\n")

    # Plot results
    t_vec = np.arange(ctrl_i) * CTRL_DT
    # plot_swing_foot_traj(t_vec, foot_traj, False)
    # plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
    # plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)

    # Replay simulation
    time_log_render = np.asarray(time_log_render, dtype=float)
    q_log_render = np.asarray(q_log_render, dtype=float)
    tau_log_render = np.asarray(tau_log_render, dtype=float)

    mujoco_go2.replay_simulation(time_log_render, q_log_render, tau_log_render, RENDER_DT, REALTIME_FACTOR)
    hold_until_all_fig_closed()
