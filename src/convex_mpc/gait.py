import numpy as np
from .go2_robot_data import PinGo2Model

# --------------------------------------------------------------------------------
# Gait Setting
# --------------------------------------------------------------------------------

PHASE_OFFSET = np.array([0, 0.25, 0.5, 0.75]).reshape(4)    # trotting gait
HEIGHT_SWING = 0.1 # Height of the swing leg trajectory apex


class Gait():
    def __init__(self, frequency_hz, duty):
        self.gait_duty = duty
        self.gait_hz = frequency_hz

        self.gait_period = 1 / frequency_hz # Perioid
        self.stance_time = self.gait_duty * self.gait_period
        self.swing_time = (1-self.gait_duty) * self.gait_period

    def compute_current_mask(self, time):

        return self.compute_contact_table(time, 0.0, 1).reshape(-1)
    
    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:

        # times: (N,)
        t = t0 + np.arange(N) * dt
        t = t + dt/2

        # phases: (4,N)
        phases = np.mod(PHASE_OFFSET[:, None] + t[None, :] / self.gait_period, 1.0)

        # mask: (4,N) with 1=stance, 0=swing
        contact_table = (phases < self.gait_duty).astype(np.int32)
        return contact_table        
    

    def compute_touchdown_world_for_traj_purpose_only(self, go2:PinGo2Model, leg:str, time_now):
        base_pos = go2.current_config.base_pos
        base_vel = go2.current_config.base_vel
        R_z = go2.R_z
        yaw_rate = go2.yaw_rate_des_world

        hip_offset = go2.get_hip_offset(leg)
        body_pos = np.array([base_pos[0], base_pos[1], 0])
        hip_pos_world = body_pos + R_z @ hip_offset

        t_swing = self.swing_time
        t_stance = self.stance_time

        # We are planning at takeoff
        T = t_swing + 0.5*t_stance
        pred_time = T / 2.0

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [base_vel[0] * pred_time, base_vel[1] * pred_time, 0]

        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  # or base_pos[0:2]
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
                                -dtheta * r_xy[1],
                                dtheta * r_xy[0],
                                0.0
                                ])
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(rotation_correction_term)
                                )
        terrain = getattr(go2, "terrain", None)
        if terrain is not None:
            pos_touchdown_world = self.select_foothold(go2, leg, pos_touchdown_world, terrain, time_now)
        return pos_touchdown_world
    

    def compute_swing_traj_and_touchdown(self, go2:PinGo2Model, leg:str, time_now):

        # This function should only be called the moment the foot takes off
        base_pos = go2.current_config.base_pos
        pos_com_world = go2.pos_com_world

        vel_com_world = go2.vel_com_world
        R_z = go2.R_z
        yaw_rate = go2.yaw_rate_des_world

        hip_offset = go2.get_hip_offset(leg)
        [foot_pos, foot_vel] = go2.get_single_foot_state_in_world(leg)
        body_pos = np.array([base_pos[0], base_pos[1], 0])


        hip_pos_world = body_pos + R_z @ hip_offset

        x_vel_des = go2.x_vel_des_world
        y_vel_des = go2.y_vel_des_world
        x_pos_des = go2.x_pos_des_world
        y_pos_des = go2.y_pos_des_world

        t_swing = self.swing_time
        t_stance = self.stance_time

        T = t_swing + 0.5*t_stance
        pred_time = T / 2.0

        # Forward (x) direction
        k_v_x = 0.4 * T          # ~0.2–0.3
        k_p_x = 0.1              # small

        # Lateral (y) direction – usually weaker
        k_v_y = 0.2 * T          # ~0.1
        k_p_y = 0.05

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [x_vel_des * pred_time, y_vel_des * pred_time, 0]
        pos_correction_term = [k_p_x * (pos_com_world[0] - x_pos_des), k_p_y * (pos_com_world[1] - y_pos_des), 0]
        vel_correction_term = [k_v_x * (vel_com_world[0] - x_vel_des), k_v_y * (vel_com_world[1] - y_vel_des), 0]
        
        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  # or base_pos[0:2]
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
                                -dtheta * r_xy[1],
                                dtheta * r_xy[0],
                                0.0
                                ])
    
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(pos_correction_term)
                                + np.array(vel_correction_term)
                                + np.array(rotation_correction_term)
                                )
        
        # Refine touchdown with terrain BEFORE building swing trajectory (Bug 2)
        terrain = getattr(go2, "terrain", None)
        if terrain is not None:
            pos_touchdown_world = self.select_foothold(go2, leg, pos_touchdown_world, terrain, time_now)

        pos_foot_traj_eval_at_world = self.make_swing_trajectory(foot_pos, pos_touchdown_world, t_swing, h_sw=HEIGHT_SWING)

        return pos_foot_traj_eval_at_world, pos_touchdown_world


    def make_swing_trajectory(self, p0, pf, t_swing, h_sw):

        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)
        T = float(t_swing)
        dp = pf - p0

        def eval_at(t):
            # phase s in [0,1]
            s = np.clip(t / T, 0.0, 1.0)

            # Minimum-jerk basis and its derivatives
            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4           # d(mj)/ds
            d2mj = 60*s    - 180*s**2 + 120*s**3         # d^2(mj)/ds^2

            # Base (x,y,z) trajectory
            p = p0 + dp * mj
            v = (dp * dmj) / T
            a = (dp * d2mj) / (T**2)

            # Optional smooth z-bump: b(s)=64*s^3*(1-s)^3, with zero vel/acc at ends
            if h_sw != 0.0:
                b    = 64 * s**3 * (1 - s)**3
                db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)           # db/ds
                d2b  = 192 * ( 2*s*(1 - s)**2*(1 - 2*s)
                            - 2*s**2*(1 - s)*(1 - 2*s)
                            - 2*s**2*(1 - s)**2 )                  # d^2b/ds^2

                p[2] += h_sw * b
                v[2] += h_sw * db / T
                a[2] += h_sw * d2b / (T**2)

            return p, v, a

        return eval_at
    # -----------------------------
    # Foothold selection constants
    # -----------------------------
    FH_R_SAMPLE = 0.15          # meters
    FH_K = 24                   # number of candidates
    FH_MAX_SLOPE_DEG = 25.0
    FH_PLANE_RES_MAX = 0.015    # meters (1.5 cm)
    FH_REACH_XY_MIN = 0.08
    FH_REACH_XY_MAX = 0.35
    FH_DZ_MAX = 0.10

    # Weights (normalized terms ~0..1)
    W_STAB = 0.20
    W_DIST = 0.25
    W_SLOPE = 0.20
    W_RESID = 0.20
    W_SPEED = 0.10
    W_GRADE = 0.05


    FH_STAB_MARGIN_MIN = 0.00   # gate: must be inside polygon
    FH_STAB_TARGET     = 0.06   # 6 cm: "good" margin

    def _project_to_plane2(self, p, origin, t1, t2):
        d = p - origin
        return np.array([np.dot(t1, d), np.dot(t2, d)], dtype=float)
    
    def _tangent_basis_from_normal(self, n):
        n = np.asarray(n, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-9)

        # pick arbitrary vector not parallel to n
        if abs(n[2]) < 0.9:
            a = np.array([0.0, 0.0, 1.0])
        else:
            a = np.array([1.0, 0.0, 0.0])

        t1 = np.cross(a, n)
        t1 = t1 / (np.linalg.norm(t1) + 1e-9)

        t2 = np.cross(n, t1)
        t2 = t2 / (np.linalg.norm(t2) + 1e-9)

        return t1, t2, n


    def _convex_hull_2d(self, pts):
        # pts: (M,2), M up to 4
        pts = np.unique(pts, axis=0)
        if len(pts) <= 2:
            return pts

        # sort by x then y
        pts = pts[np.lexsort((pts[:,1], pts[:,0]))]

        def cross(o,a,b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in pts[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = np.array(lower[:-1] + upper[:-1], dtype=float)
        return hull

    def _point_margin_to_poly(self, poly, q):
        """
        poly: (H,2) convex hull vertices CCW-ish
        q: (2,)
        Returns signed margin:
        >0 inside: min distance to edges
        <0 outside: -min distance to edges
        """
        H = poly.shape[0]
        if H < 3:
            return -np.inf

        inside = True
        min_dist = np.inf

        for i in range(H):
            a = poly[i]
            b = poly[(i+1) % H]
            e = b - a
            # outward normal for CCW polygon is [-e_y, e_x] (but sign depends on ordering)
            # We'll compute signed distance using cross test:
            # For CCW, cross(e, q-a) >= 0 means q is left of edge (inside)
            cross_val = e[0]*(q[1]-a[1]) - e[1]*(q[0]-a[0])
            if cross_val < 0:
                inside = False

            # distance from point to segment
            t = np.dot(q - a, e) / (np.dot(e, e) + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * e
            d = np.linalg.norm(q - proj)
            min_dist = min(min_dist, d)

        return min_dist if inside else -min_dist


    def _unit2(self, v, eps=1e-9):
        n = np.linalg.norm(v)
        return v / (n + eps), n

    def _plane_fit_residual(self, terrain, x, y, r_patch=0.06, grid=5):
        """
        Plane fit residual on a heightmap-like surface.
        Uses only terrain.height_and_normal(x,y) for z sampling.
        Returns RMS residual (meters). Low = locally planar.
        """
        xs = np.linspace(x - r_patch, x + r_patch, grid)
        ys = np.linspace(y - r_patch, y + r_patch, grid)

        pts = []
        for xi in xs:
            for yi in ys:
                zi, _ = terrain.height_and_normal(xi, yi)
                pts.append([xi, yi, zi])
        pts = np.array(pts, dtype=float)

        # Fit z = ax + by + c via least squares
        A = np.c_[pts[:,0], pts[:,1], np.ones(len(pts))]
        z = pts[:,2]
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        z_hat = A @ coef
        resid = z - z_hat
        rms = np.sqrt(np.mean(resid**2))
        return float(rms)

    def _candidate_offsets(self, r):
        """
        Structured sampling: center + inner ring + outer ring + forward-biased.
        Total ~24.
        """
        offs = []
        offs.append([0.0, 0.0])

        # inner ring (8)
        for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
            offs.append([0.5*r*np.cos(a), 0.5*r*np.sin(a)])

        # outer ring (12)
        for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
            offs.append([r*np.cos(a), r*np.sin(a)])

        # 3 extra "forward-ish" slots (we'll rotate these into v_dir later)
        offs.append([0.8*r, 0.0])
        offs.append([0.6*r, 0.15*r])
        offs.append([0.6*r, -0.15*r])

        return np.array(offs, dtype=float)  # (K,2)

    def select_foothold(self, go2, leg: str, nominal_td_world: np.ndarray, terrain, time_now):
        """
        Returns chosen touchdown position in world: (3,)
        Assumes terrain.height_and_normal(x,y) exists.
        """
        td0 = np.array(nominal_td_world, dtype=float)

        # --- walking direction (world XY) ---
        v_des_xy = np.array([getattr(go2, "x_vel_des_world", 0.0),
                             getattr(go2, "y_vel_des_world", 0.0)], dtype=float)
        v_dir, v_norm = self._unit2(v_des_xy)

        if v_norm < 1e-3:
            # fallback to heading from R_z
            fwd = go2.R_z @ np.array([1.0, 0.0, 0.0])
            v_dir, _ = self._unit2(fwd[:2])

        # build rotation that maps +x (local) -> v_dir (world)
        # world_offset = [v_dir, v_perp] @ local_offset
        v_perp = np.array([-v_dir[1], v_dir[0]], dtype=float)
        R2 = np.c_[v_dir, v_perp]  # 2x2

        # --- sample candidates ---
        offs_local = self._candidate_offsets(self.FH_R_SAMPLE)
        offs_world = (R2 @ offs_local.T).T  # (K,2)

        # hip position for reachability gate
        hip_offset = go2.get_hip_offset(leg)
        base_pos = go2.current_config.base_pos
        hip_pos_world = np.array([base_pos[0], base_pos[1], 0.0]) + go2.R_z @ hip_offset
        hip_xy = hip_pos_world[:2]

        # estimate "grade direction" using heightmap gradient approx along v_dir
        def height_at(x,y):
            z,_ = terrain.height_and_normal(x,y)
            return float(z)

        def grade_along_dir(x,y,dir2,eps=0.05):
            z1 = height_at(x + eps*dir2[0], y + eps*dir2[1])
            z0 = height_at(x - eps*dir2[0], y - eps*dir2[1])
            return (z1 - z0) / (2*eps)  # positive = uphill along dir

        # leg preference: front feet prefer uphill, rear prefer downhill
        is_front = leg in ("FL","FR")
        prefer_sign = +1.0 if is_front else -1.0

        # collect normalized terms to compute robustly
        cand = []
        terms = {
            "dist": [],
            "slope": [],
            "resid": [],
            "speed": [],
            "grade": [],
            "stab": []
        }


        # precompute nominal z
        z0, _ = terrain.height_and_normal(td0[0], td0[1])

        for i in range(min(self.FH_K, len(offs_world))):
            xy = td0[:2] + offs_world[i]
            x, y = float(xy[0]), float(xy[1])

            z, n = terrain.height_and_normal(x, y)
            n = np.asarray(n, dtype=float)
            n = n / (np.linalg.norm(n) + 1e-9)

            # ---- Hard gates ----
            # slope
            slope_deg = np.degrees(np.arctan2(np.linalg.norm(n[:2]), n[2]))
            if slope_deg > self.FH_MAX_SLOPE_DEG:
                continue

            # reachability in XY
            dxy = np.linalg.norm(xy - hip_xy)
            if dxy < self.FH_REACH_XY_MIN or dxy > self.FH_REACH_XY_MAX:
                continue

            # z jump limit
            if abs(float(z) - float(z0)) > self.FH_DZ_MAX:
                continue

            # plane-fit residual (curvature / hole / rim awareness)
            resid = self._plane_fit_residual(terrain, x, y)
            if resid > self.FH_PLANE_RES_MAX:
                continue
            
            # --- stance set at touchdown time ---
            t_td = time_now + self.swing_time
            mask_td = self.compute_current_mask(t_td)  # (4,)

            # build support points list in WORLD
            legs = ["FL","FR","RL","RR"]
            support_pts = []
            support_normals = []

            for j, lj in enumerate(legs):
                if mask_td[j] != 1:
                    continue

                if lj == leg:
                    # candidate foothold
                    support_pts.append(np.array([x, y, z], dtype=float))
                    support_normals.append(n)
                else:
                    # use current foot position as approximation of stance contact
                    pj, _ = go2.get_single_foot_state_in_world(lj)
                    # snap it to terrain (important)
                    zj, nj = terrain.height_and_normal(pj[0], pj[1])
                    pj = np.array([pj[0], pj[1], zj], dtype=float)
                    nj = np.asarray(nj, dtype=float)
                    nj = nj / (np.linalg.norm(nj) + 1e-9)
                    support_pts.append(pj)
                    support_normals.append(nj)

            # need at least 3 contacts
            if len(support_pts) < 3:
                continue

            support_pts = np.array(support_pts, dtype=float)
            n_sup = np.mean(np.array(support_normals), axis=0)
            n_sup = n_sup / (np.linalg.norm(n_sup) + 1e-9)

            # tangent basis from n_sup
            t1, t2, _ = self._tangent_basis_from_normal(n_sup)

            origin = support_pts[0]
            pts2 = np.array([self._project_to_plane2(p, origin, t1, t2) for p in support_pts])
            poly = self._convex_hull_2d(pts2)

            # COM point for stability check
            p_com = go2.pos_com_world + go2.vel_com_world * self.swing_time
            q2 = self._project_to_plane2(p_com, origin, t1, t2)

            margin = self._point_margin_to_poly(poly, q2)

            # ---- stability gate ----
            if margin < self.FH_STAB_MARGIN_MIN:
                continue

            # ---- soft term: larger margin is better ----
            # convert to penalty where "good" margin -> 0 penalty
            stab_pen = max(0.0, (self.FH_STAB_TARGET - margin) / (self.FH_STAB_TARGET + 1e-9))
            terms["stab"].append(stab_pen)

            # ---- Soft terms (later normalized) ----
            dist = np.linalg.norm(xy - td0[:2])  # deviation from nominal

            # speed bias: prefer forward along v_dir
            forward = np.dot((xy - td0[:2]), v_dir)  # >0 means ahead

            # grade preference: front likes uphill, rear likes downhill
            g = grade_along_dir(x, y, v_dir)
            grade_term = -prefer_sign * g  # lower is better (preferred sign reduces score)

            cand.append((x, y, float(z), n))
            terms["dist"].append(dist)
            terms["slope"].append(slope_deg)
            terms["resid"].append(resid)
            terms["speed"].append(-forward)     # lower is better (more forward => smaller)
            terms["grade"].append(grade_term)   # lower is better (meets preference)

        # If everything got gated out, return nominal snapped-to-terrain
        if len(cand) == 0:
            z, n = terrain.height_and_normal(td0[0], td0[1])
            td = td0.copy()
            td[2] = z
            return td

        # ---- normalize helper (0..1) ----
        def norm01(arr, eps=1e-9):
            a = np.array(arr, dtype=float)
            lo, hi = np.min(a), np.max(a)
            if hi - lo < eps:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo)

        ndist  = norm01(terms["dist"])
        nslope = norm01(terms["slope"])
        nres   = norm01(terms["resid"])
        nspd   = norm01(terms["speed"])
        ngr    = norm01(terms["grade"])
        nstab = norm01(terms["stab"])

        # ---- total score (lower is better) ----
        score = (self.W_DIST  * ndist +
                 self.W_SLOPE * nslope +
                 self.W_RESID * nres +
                 self.W_SPEED * nspd +
                 self.W_GRADE * ngr +
                 self.W_STAB * nstab)

        best = int(np.argmin(score))
        x, y, z, n = cand[best]
        return np.array([x, y, z], dtype=float)


