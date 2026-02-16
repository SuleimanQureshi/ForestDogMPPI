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
RUN_SIM_LENGTH_S = 10.0

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
@dataclass
class MuJoCoLidar3D:
    """
    Very simple 3D LiDAR using MuJoCo ray casts.
    Returns hit points in WORLD frame.
    """
    def __init__(self, model, data,
                 n_az=72, n_el=5,
                 el_min_deg=-10.0, el_max_deg=10.0,
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

    def scan(self, sensor_pos_world, yaw):
        """
        sensor_pos_world: (3,)
        yaw: float
        """
        # rotate body directions into world with yaw only (good enough for now)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)

        dirs_world = (Rz @ self.dirs_body.T).T  # (M,3)

        hits = []
        p0 = np.asarray(sensor_pos_world, dtype=float)

        # optional: exclude nothing (you can later exclude robot geoms with geomgroup)
        geomgroup = np.ones(6, dtype=np.uint8)

        for d in dirs_world:
            geomid = np.array([-1], dtype=np.int32)

            dist = mj.mj_ray(
                self.model,
                self.data,
                p0,
                d,
                geomgroup,
                1,          # include static geoms
                -1,         # bodyexclude
                geomid      # <-- REQUIRED in your version
            )

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
        self.H = 80              # ~1.66s lookahead with dt=0.0208
        self.BATCH = 600         # still feasible
        self.ITERS = 3           # better convergence

        self.LAMBDA = 1.0
        self.ALPHA = 0.85        # correlated noise

        # velocity limits (keep conservative!)
        self.vx_min, self.vx_max = 0.0, 0.6
        self.vy_min, self.vy_max = -0.3, 0.3
        self.wz_min, self.wz_max = -1.0, 1.0

        # noise std
        self.std = np.array([0.12, 0.10, 0.30])  # allow lateral/turn exploration

        # persistent sequence
        self.U = np.zeros((self.H, 3))

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
        return eps

    # --------------------------------------
    # rollout model
    # --------------------------------------
    def rollout(self, state, U_batch):

        B, T, _ = U_batch.shape
        X = np.zeros((B, T, 3))

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

            X[:, t, 0] = x
            X[:, t, 1] = y
            X[:, t, 2] = yaw

        return X

    # --------------------------------------
    # cost
    # --------------------------------------
    def cost(self, X, U_batch, goal, obstacle_xy):


        x = X[:, :, 0]
        y = X[:, :, 1]
        yaw = X[:, :, 2]

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
        if obstacle_xy is not None and obstacle_xy.shape[0] > 0:

            px = x[:, :, None]
            py = y[:, :, None]

            ox = obstacle_xy[:, 0][None, None, :]
            oy = obstacle_xy[:, 1][None, None, :]

            d = np.sqrt((px - ox)**2 + (py - oy)**2)
            dmin = d.min(axis=2)

            robot_radius = 0.35
            safety_margin = 0.4
            inflation = robot_radius + safety_margin

            inside = np.maximum(0.0, inflation - dmin)
            hard = (inside**2).mean(axis=1)

            soft = np.exp(-3.0 * dmin).mean(axis=1)

            obs_cost = 500.0 * hard + 20.0 * soft

        else:
            obs_cost = np.zeros(x.shape[0])

        


        #Progress param
        d0 = np.sqrt((x[:, 0] - goal[0])**2 + (y[:, 0] - goal[1])**2)      # distance at start of rollout
        dT = np.sqrt((x[:, -1] - goal[0])**2 + (y[:, -1] - goal[1])**2)    # distance at end
        progress = (d0 - dT)  # positive = moved toward goal


        # smoothness
        dU = np.diff(U_batch, axis=1)
        smooth = (dU**2).mean(axis=(1,2))

        # control effort
        effort = (U_batch**2).mean(axis=(1,2))

        return (
            4.0 * goal_cost +
            10.0 * term +
            3.0 * heading_cost +
            1.0 * obs_cost +
            0.2 * smooth +
            0.05 * effort
            - 6.0 * progress
        )

    # --------------------------------------
    # main step
    # --------------------------------------
    def command(self, state, goal, obstacle_xy):
        
        for _ in range(self.ITERS):

            eps = self.correlated_noise()
            U_batch = self.U[None, :, :] + eps

            X = self.rollout(state, U_batch)
            costs = self.cost(X, U_batch, goal, obstacle_xy)

            beta = costs.min()
            w = np.exp(-(costs - beta) / self.LAMBDA)
            w /= (w.sum() + 1e-9)

            dU = (w[:, None, None] * eps).sum(axis=0)
            self.U += dU

            # clip feasible
            self.U[:, 0] = np.clip(self.U[:, 0], self.vx_min, self.vx_max)
            self.U[:, 1] = np.clip(self.U[:, 1], self.vy_min, self.vy_max)
            self.U[:, 2] = np.clip(self.U[:, 2], self.wz_min, self.wz_max)

        # execute first control
        u0 = self.U[0].copy()

        # receding horizon
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]

        self.last_U_batch = U_batch

        return u0


import matplotlib.pyplot as plt

def debug_plot_mppi(state, goal, obstacle_xy, U_batch, mppi):

    plt.clf()

    # --- Rollout all trajectories ---
    X = mppi.rollout(state, U_batch)

    for i in range(min(80, X.shape[0])):  # don't plot all 600
        plt.plot(X[i,:,0], X[i,:,1], color='blue', alpha=0.1)

    # --- Robot position ---
    plt.scatter(state[0], state[1], c='black', s=80, label="Robot")

    # --- Goal ---
    plt.scatter(goal[0], goal[1], c='green', s=120, label="Goal")

    # --- Obstacles (LiDAR points) ---
    if obstacle_xy is not None and obstacle_xy.shape[0] > 0:
        plt.scatter(obstacle_xy[:,0], obstacle_xy[:,1], c='red', s=5)

    plt.axis("equal")
    plt.xlim(-4,4)
    plt.ylim(-4, 4)
    plt.legend()
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

# --------------------------------------------------------------------------------
# Simulation Initialization
# --------------------------------------------------------------------------------

go2 = PinGo2Model()
mujoco_go2 = MuJoCo_GO2_Model()
lidar = MuJoCoLidar3D(mujoco_go2.model, mujoco_go2.data, n_az=72, n_el=5, max_range=6.0)
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

goal_xy = np.array([3.0, 0.0])
box_radius = 0.75

# Initialize robot configuration
q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_go2.update_with_q_pin(q_init)

# Set physics dt (keep it fast!)
mujoco_go2.model.opt.timestep = SIM_DT


# Safe defaults until first solve
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
                    lidar_origin = np.array([px, py, 0.35], dtype=float)

                    hits_world = lidar.scan(lidar_origin, yaw)        # (N,3)
                    obstacle_xy = hits_world[:, :2]                   # (N,2)

                    # downsample for speed (VERY important)
                    if obstacle_xy.shape[0] > 250:
                        idx = np.random.choice(obstacle_xy.shape[0], 250, replace=False)
                        obstacle_xy = obstacle_xy[idx]

                    u0 = mppi.command(state0, goal_xy, obstacle_xy)
                if ctrl_i % 10 == 0:  # don’t draw every tick
                    debug_plot_mppi(
                        state0,
                        goal_xy,
                        obstacle_xy,
                        mppi.last_U_batch,
                        mppi
                    )


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
        viewer.sync()
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
