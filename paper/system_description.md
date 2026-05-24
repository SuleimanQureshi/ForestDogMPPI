# Integrated Terrain-Aware Navigation and Locomotion for Quadruped Robots using Hierarchical MPC

## II. SYSTEM ARCHITECTURE

We present a tightly-integrated locomotion and navigation framework for the Unitree Go2 quadruped robot operating on unstructured terrain. The key insight is that a single terrain representation---a dual-layer heightmap constructed from onboard LiDAR---simultaneously informs decisions at every level of the control hierarchy: global path planning, local trajectory optimization, contact force generation, foothold selection, and swing leg control.

Fig. 1 illustrates the data flow. The system operates as a cascaded loop at four rates:

| Layer | Rate | Function |
|-------|------|----------|
| Physics simulation | 1000 Hz | MuJoCo rigid-body dynamics |
| Leg controller | 200 Hz | Cartesian impedance + force tracking |
| Centroidal MPC | 30--50 Hz | Convex QP for optimal contact forces |
| MPPI planner | 15--25 Hz | Sampling-based velocity command generation |
| Perception + A* | ~6 Hz | LiDAR scan, heightmap update, global replan |

The terrain representation is consumed by five distinct subsystems (indicated by dashed arrows in Fig. 1):
1. **Friction cone rotation** in the MPC (Section II-D),
2. **COM height adaptation** in the trajectory generator (Section II-E),
3. **Foothold quality scoring** in the gait planner (Section II-F),
4. **Steppability and obstacle costs** in the MPPI local planner (Section II-C),
5. **Edge traversability** in the A* global planner (Section II-B).

This coupling ensures that the robot's navigation decisions are grounded in the same physical constraints that govern its locomotion.

---

## III. TERRAIN PERCEPTION AND REPRESENTATION

### A. 3D LiDAR Simulation

We simulate a 3D LiDAR sensor mounted at the robot's trunk using MuJoCo ray-casting. The sensor fires $N_{az} \times N_{el} = 90 \times 15 = 1350$ rays per scan, spanning azimuth $[-\pi, \pi)$ and elevation $[-30^\circ, 15^\circ]$ with a maximum range of 6 m. Each ray direction in the body frame is

$$\mathbf{d}_{\text{body}} = \begin{bmatrix} \cos\phi\cos\theta \\ \cos\phi\sin\theta \\ \sin\phi \end{bmatrix}$$

where $\theta$ is azimuth and $\phi$ is elevation. Rays are rotated to world frame via the trunk orientation $\mathbf{R}_{\text{wb}}$, and hits on the robot's own body are discarded. Points with radial distance $< 0.45$ m from the base (self-hits near feet) or with height outside $[-0.5, 2.0]$ m are filtered.

### B. Dual-Layer Global Heightmap

We maintain a global heightmap $\mathcal{H}$ on a 2D grid of resolution $\delta = 0.05$ m covering a $12 \times 12$ m region. Each cell $(i,j)$ stores two height values updated via exponential moving average (EMA):

**Ground layer** (walkable surface estimate):
$$h^{\text{ground}}_{i,j} \leftarrow (1 - \alpha_g)\, h^{\text{ground}}_{i,j} + \alpha_g\, Q_{0.2}(\{z_k\}_{k \in \text{cell}})$$

where $Q_{0.2}$ denotes the 20th percentile of height samples within the cell, and $\alpha_g = 0.25$.

**Top layer** (highest observed surface):
$$h^{\text{top}}_{i,j} \leftarrow (1 - \alpha_t)\, h^{\text{top}}_{i,j} + \alpha_t\, \max(\{z_k\}_{k \in \text{cell}})$$

with $\alpha_t = 0.50$. The ground layer uses a low quantile to robustly estimate the walkable surface even when obstacle points (e.g., tree trunks) fall in the same cell. The top layer captures obstacle extent.

### C. Derived Terrain Quantities

From the dual-layer heightmap we derive three quantities consumed by downstream modules:

**Surface normal.** Estimated via central differences on the ground layer:

$$\hat{\mathbf{n}}_{i,j} = \frac{1}{\|\mathbf{n}\|}\begin{bmatrix} -\partial h^{\text{ground}} / \partial x \\ -\partial h^{\text{ground}} / \partial y \\ 1 \end{bmatrix}, \quad \frac{\partial h^{\text{ground}}}{\partial x} \approx \frac{h^{\text{ground}}_{i,j+1} - h^{\text{ground}}_{i,j-1}}{2\delta}$$

**Clearance.** Per-cell vertical gap between top and ground layers:

$$c_{i,j} = h^{\text{top}}_{i,j} - h^{\text{ground}}_{i,j}$$

Cells with $c_{i,j} \geq c_{\text{lethal}} = 0.25$ m are classified as obstacles.

**Steppability (Foothold Quality).** A composite terrain quality metric $\sigma \in [0, 1]$ (0 = ideal, 1 = untraversable):

$$\sigma(\mathbf{p}) = 0.6\, \sigma_{\text{slope}}(\mathbf{p}) + 0.4\, \sigma_{\text{rough}}(\mathbf{p})$$

where

$$\sigma_{\text{slope}} = \text{clip}\!\left(\frac{\|\nabla h^{\text{ground}}\|}{0.5},\; 0,\; 1\right), \qquad \sigma_{\text{rough}} = \text{clip}\!\left(\frac{\text{std}_5(\mathbf{p})}{0.03},\; 0,\; 1\right)$$

Here $\text{std}_5$ is the standard deviation of ground heights over a 5-point cross pattern (center + 4 cardinal neighbors at spacing $\delta$). A slope magnitude of 0.5 ($\approx 27^\circ$) saturates the slope term; surface roughness of 3 cm saturates the roughness term.

### D. Obstacle Cost Map

Obstacle cells are inflated radially to create a smooth cost field for planning:

$$\mathcal{C}(\mathbf{p}) = \max_{(i,j) \in \mathcal{O}} \left[ \max\!\left(0,\; 1 - \frac{\|\mathbf{p} - \mathbf{p}_{ij}\|}{r_{\text{inflate}}}\right) \right]$$

where $\mathcal{O}$ is the set of lethal cells and $r_{\text{inflate}} = 0.70$ m. The cost map applies temporal decay $\mathcal{C} \leftarrow 0.98\,\mathcal{C}$ each update cycle, allowing previously-observed obstacles that leave the sensor field to fade.

---

## IV. GLOBAL PATH PLANNING

### A. Terrain-Aware Weighted A*

Given the robot's current position $\mathbf{p}_{\text{start}}$ and a goal $\mathbf{p}_{\text{goal}}$, we compute a global path on the cost map grid using weighted A*. The edge cost between adjacent cells incorporates both obstacle proximity and terrain traversability:

$$w(i \to j) = \|\mathbf{p}_i - \mathbf{p}_j\| \cdot \left(1 + w_{\text{obs}}\left(e^{4\,\mathcal{C}_j} - 1\right) + w_{\text{terr}}\,\sigma_j\right)$$

where $w_{\text{obs}} = 100$, $w_{\text{terr}} = 5$, $\mathcal{C}_j$ is the obstacle cost, and $\sigma_j$ is the steppability score at cell $j$. The exponential term $e^{4\mathcal{C}} - 1$ creates a steep penalty near obstacles so that A* strongly prefers clearance over shorter paths. Cells with $\mathcal{C} \geq 0.8$ are treated as impassable.

The heuristic is $h(\mathbf{p}) = 1.5\,\|\mathbf{p} - \mathbf{p}_{\text{goal}}\|$ (weighted A* with $\epsilon = 1.5$ for faster planning at bounded suboptimality).

### B. Path Hysteresis

To prevent left-right flip-flopping on successive replans around obstacles, cells lying on the previous path receive a 50% edge-cost discount:

$$w_{\text{hysteresis}}(i \to j) = \begin{cases} 0.5\, w(i \to j) & \text{if } j \in \mathcal{P}_{\text{prev}} \\ w(i \to j) & \text{otherwise} \end{cases}$$

### C. Path Post-Processing

The raw A* path is resampled to uniform arc-length spacing of 0.15 m, then smoothed via 40 iterations of Laplacian averaging with collision checking:

$$\mathbf{p}_k \leftarrow (1 - \alpha)\,\mathbf{p}_k + \frac{\alpha}{2}(\mathbf{p}_{k-1} + \mathbf{p}_{k+1}), \quad \alpha = 0.25$$

Each smoothed candidate is accepted only if $\mathcal{C}(\mathbf{p}_k) < 0.8$; otherwise the original point is retained. Start and end points are pinned.

---

## V. LOCAL TRAJECTORY OPTIMIZATION (MPPI)

### A. Dynamics Model

The MPPI local planner generates body-frame velocity commands $\mathbf{u} = [v_x, v_y, \omega_z]^T$ by optimizing over a finite horizon. We model the robot as a planar unicycle with first-order velocity dynamics:

$$\dot{x} = v_x \cos\psi - v_y \sin\psi, \quad \dot{y} = v_x \sin\psi + v_y \cos\psi, \quad \dot{\psi} = \omega_z$$

$$\dot{v}_x = \frac{v_x^{\text{cmd}} - v_x}{\tau}, \quad \dot{v}_y = \frac{v_y^{\text{cmd}} - v_y}{\tau}, \quad \dot{\omega}_z = \frac{\omega_z^{\text{cmd}} - \omega_z}{\tau}$$

where $\tau = 0.4$ s is a velocity tracking time constant that approximates the closed-loop response of the lower-level MPC and leg controller. The state is $\mathbf{s} = [x, y, \psi, v_x, v_y, \omega_z]^T \in \mathbb{R}^6$.

### B. Sampling and Optimization

At each MPPI cycle, we maintain a nominal control sequence $\mathbf{U} = [\mathbf{u}_0, \ldots, \mathbf{u}_{H-1}] \in \mathbb{R}^{H \times 3}$ with horizon $H = 80$ steps. We draw $K = 400$ perturbation sequences with temporally-correlated AR(1) noise:

$$\boldsymbol{\epsilon}^{(k)}_t = \alpha\,\boldsymbol{\epsilon}^{(k)}_{t-1} + (1 - \alpha)\,\boldsymbol{\xi}^{(k)}_t, \quad \boldsymbol{\xi}^{(k)}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\alpha = 0.5$ and $\boldsymbol{\Sigma} = \text{diag}(\sigma_{v_x}^2, \sigma_{v_y}^2, \sigma_{\omega_z}^2)$. The noise standard deviations are adapted online (Section V-D). Candidate control sequences are $\mathbf{U}^{(k)} = \mathbf{U} + \boldsymbol{\epsilon}^{(k)}$.

After rolling out all $K$ trajectories, the nominal sequence is updated via importance-weighted averaging:

$$\mathbf{U} \leftarrow \sum_{k=1}^{K} w_k\, \mathbf{U}^{(k)}, \quad w_k = \frac{\exp\left(-\frac{1}{\lambda}(J_k - J_{\min})\right)}{\sum_{j=1}^{K}\exp\left(-\frac{1}{\lambda}(J_j - J_{\min})\right)}$$

with temperature $\lambda = 8.0$. This is repeated for $I = 5$ iterations per cycle. After extracting the first command $\mathbf{u}_0$, the sequence is shifted forward (receding horizon) with the last element duplicated.

### C. Cost Function (Nav2-Style Critics)

The total cost for rollout $k$ is a sum of six critic terms:

$$J_k = J_{\text{path}} + J_{\text{heading}} + J_{\text{turn}} + J_{\text{progress}} + J_{\text{obs}} + J_{\text{step}} + J_{\text{slope}} + J_{\text{smooth}}$$

**1) Path-Following Critic.** Penalizes cross-track error to the global path $\mathcal{P}$. For subsampled timesteps $\{t_s\}$:

$$J_{\text{path}} = 80 \cdot \frac{1}{|\{t_s\}|}\sum_{t_s} \min_{\mathbf{p}_j \in \mathcal{P}} \|\mathbf{x}^{(k)}_{t_s} - \mathbf{p}_j\|$$

**2) Path Angle Critic.** Aligns the robot heading $\psi$ with the path tangent $\hat{\mathbf{t}}$ at the nearest path point:

$$J_{\text{heading}} = 120 \cdot \frac{1}{|\{t_s\}|}\sum_{t_s} \left|\text{atan2}\!\left(\sin(\psi_{t_s} - \theta_{\hat{\mathbf{t}}}),\; \cos(\psi_{t_s} - \theta_{\hat{\mathbf{t}}})\right)\right|$$

where $\theta_{\hat{\mathbf{t}}} = \text{atan2}(\hat{t}_y, \hat{t}_x)$ is the tangent heading.

**3) Turn-in-Place Critic.** Prevents forward motion when the heading error is large, avoiding unstable sideways ("crab-walk") locomotion:

$$J_{\text{turn}} = 500 \cdot \frac{1}{|\{t_s\}|}\sum_{t_s} \left[\max(v_{x,t_s}, 0)\right]^2 \cdot \left[\max(|e_{\psi,t_s}| - 0.2, 0)\right]^2$$

This critic is crucial for legged robots: unlike wheeled platforms, a quadruped cannot safely translate laterally at speed. The critic teaches MPPI to stop, turn to face the path tangent, then proceed forward.

**4) Path Progress Critic.** Rewards rollouts that reach further along the path arc length:

$$J_{\text{progress}} = 20 \cdot \frac{L_{\text{total}} - s_{\max}^{(k)}}{L_{\text{total}}}$$

where $s_{\max}^{(k)}$ is the maximum arc-length progress over all subsampled timesteps (not just the terminal state), ensuring a gradient even for slowly-moving rollouts.

**5) Obstacle Critic.** A graduated penalty using the inflated cost map $\mathcal{C}$:

$$J_{\text{obs}} = \frac{1}{20}\sum_{t=0}^{H-1} P_t + 3 \cdot \max_t P_t, \quad P_t = \begin{cases} 300 & \text{if } \mathcal{C}_t \geq 1.0 \\ 150\,\mathcal{C}_t & \text{if } \mathcal{C}_t > 0.3 \\ 30\,\mathcal{C}_t & \text{otherwise} \end{cases}$$

The $\max$ term ensures that even a single collision is heavily penalized.

**6) Foothold-Quality-Aware (FQA) Steppability Critic.** This is a novel critic specific to legged robots. For each rollout position and heading, we predict the world-frame location of all four feet using known hip offsets $\mathbf{h}_l \in \mathbb{R}^2$ (body frame):

$$\mathbf{p}_l^{\text{foot}}(t) = \begin{bmatrix} x_t \\ y_t \end{bmatrix} + \mathbf{R}(\psi_t)\,\mathbf{h}_l, \quad l \in \{\text{FL, FR, RL, RR}\}$$

where $\mathbf{R}(\psi)$ is the 2D rotation matrix. The steppability $\sigma$ (Section III-C) is queried at each predicted foothold:

$$J_{\text{step}} = \frac{w_{\text{step}}}{4} \sum_{l=1}^{4} \frac{1}{H}\sum_{t=0}^{H-1} \sigma^2(\mathbf{p}_l^{\text{foot}}(t)), \quad w_{\text{step}} = 3.0$$

This term distinguishes our formulation from standard MPPI for wheeled robots: instead of only checking center-of-mass collision, we evaluate terrain quality at the actual foot contact locations. Trajectories that route feet over steep slopes, rough surfaces, or terrain edges are penalized.

**7) Slope Critic.** Penalizes traversal of steep terrain:

$$J_{\text{slope}} = 2.5 \cdot \frac{1}{H}\sum_{t=0}^{H-1} \|\nabla h^{\text{ground}}(\mathbf{x}_t)\|$$

**8) Smoothness Critic.** Regularizes control changes:

$$J_{\text{smooth}} = 0.1 \cdot \frac{1}{H-1}\sum_{t=0}^{H-2} |\mathbf{u}_{t+1} - \mathbf{u}_t|$$

### D. Adaptive Noise and Warm-Starting

The noise standard deviation $\boldsymbol{\Sigma}$ is adapted based on context:

- **Near goal and on-track** ($d < 1.5$ m, cross-track $< 0.3$ m): $\boldsymbol{\sigma} = [0.08, 0.05, 0.40]$ (narrow, exploit)
- **Stuck or near obstacle**: $\boldsymbol{\sigma} = [0.30, 0.10, 1.20]$ (wide yaw exploration)
- **Default**: $\boldsymbol{\sigma} = [0.35, 0.05, 1.00]$

Stuck detection triggers when goal-distance progress is $< 0.1$ m over 6 consecutive MPPI calls.

When the mean forward velocity in $\mathbf{U}$ decays below 0.10 m/s (e.g., after a replan), the sequence is re-seeded from the path tangent: $v_x = 0.35$ m/s if heading error $< 0.3$ rad, else $v_x = 0$; $\omega_z = 2.5\,e_\psi$.

The first command $\mathbf{u}_0$ is acceleration-clamped before execution:

$$\mathbf{u}_0 \leftarrow \mathbf{u}_{\text{prev}} + \text{clip}(\mathbf{u}_0 - \mathbf{u}_{\text{prev}},\; -\mathbf{a}_{\max}\Delta t,\; \mathbf{a}_{\max}\Delta t)$$

with $\mathbf{a}_{\max} = [2.0, 1.5, 10.0]$ in $[\text{m/s}^2, \text{m/s}^2, \text{rad/s}^2]$.

---

## VI. CENTROIDAL MODEL PREDICTIVE CONTROL

### A. Centroidal Dynamics

We model the quadruped as a single rigid body (SRB) with mass $m$ and rotational inertia $\mathbf{I}_{\text{com}}$ expressed in the world frame. The centroidal state is

$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \boldsymbol{\Theta} \\ \dot{\mathbf{p}} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{12}$$

where $\mathbf{p} \in \mathbb{R}^3$ is the center-of-mass (COM) position, $\boldsymbol{\Theta} = [\phi, \theta, \psi]^T$ are roll-pitch-yaw Euler angles, $\dot{\mathbf{p}} \in \mathbb{R}^3$ is COM velocity, and $\boldsymbol{\omega} \in \mathbb{R}^3$ is angular velocity in the world frame.

The control input is the concatenated contact forces at four feet:

$$\mathbf{u} = \begin{bmatrix} \mathbf{f}_{\text{FL}} \\ \mathbf{f}_{\text{FR}} \\ \mathbf{f}_{\text{RL}} \\ \mathbf{f}_{\text{RR}} \end{bmatrix} \in \mathbb{R}^{12}$$

The continuous-time dynamics are:

$$\dot{\mathbf{x}} = \mathbf{A}_c\,\mathbf{x} + \mathbf{B}_c(t)\,\mathbf{u} + \mathbf{g}_c$$

with

$$\mathbf{A}_c = \begin{bmatrix} \mathbf{0} & \mathbf{0} & \mathbf{I}_3 & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{R}_z^T \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \end{bmatrix}, \quad \mathbf{B}_c(t) = \begin{bmatrix} \mathbf{0}_{3 \times 12} \\ \mathbf{0}_{3 \times 12} \\ \frac{1}{m}\begin{bmatrix}\mathbf{I}_3 & \mathbf{I}_3 & \mathbf{I}_3 & \mathbf{I}_3\end{bmatrix} \\ \mathbf{I}_{\text{com}}^{-1}\begin{bmatrix}[\mathbf{r}_1]_\times & [\mathbf{r}_2]_\times & [\mathbf{r}_3]_\times & [\mathbf{r}_4]_\times\end{bmatrix} \end{bmatrix}$$

$$\mathbf{g}_c = \begin{bmatrix} \mathbf{0}_6 \\ [0, 0, -g]^T \\ \mathbf{0}_3 \end{bmatrix}$$

Here $\mathbf{R}_z$ is the yaw-only rotation matrix, $\mathbf{r}_l = \mathbf{p}_l^{\text{foot}} - \mathbf{p}_{\text{com}}$ is the lever arm from COM to foot $l$, and $[\cdot]_\times$ denotes the skew-symmetric cross-product matrix. The matrix $\mathbf{B}_c$ is time-varying because the lever arms change as the COM moves along the reference trajectory.

### B. Discretization

We discretize over a horizon of $N = 16$ steps spanning one gait cycle $T_{\text{gait}} = 1/f_{\text{gait}}$, with timestep $\Delta t = T_{\text{gait}}/N$. The state transition uses zero-order hold (ZOH):

$$\mathbf{A}_d, \mathbf{B}_{d,k} = \text{ZOH}(\mathbf{A}_c, \mathbf{B}_{c,k}, \Delta t)$$

The discrete gravity term requires integration of the matrix exponential:

$$\mathbf{g}_d = \int_0^{\Delta t} e^{\mathbf{A}_c \tau}\,\mathbf{g}_c\,d\tau \approx \text{trapz}\!\left(\left\{e^{\mathbf{A}_c \tau_i}\,\mathbf{g}_c\right\}_{i=0}^{49},\; \tau \in [0, \Delta t]\right)$$

computed via 50-point trapezoidal quadrature.

### C. Convex QP Formulation

The MPC solves the following quadratic program at each control cycle:

$$\min_{\mathbf{x}_{1:N},\, \mathbf{u}_{0:N-1}} \sum_{k=0}^{N-1}\left[\|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_{\mathbf{Q}}^2 + \|\mathbf{u}_k\|_{\mathbf{R}}^2\right]$$

subject to:

$$\mathbf{x}_{k+1} = \mathbf{A}_d\,\mathbf{x}_k + \mathbf{B}_{d,k}\,\mathbf{u}_k + \mathbf{g}_d, \quad k = 0, \ldots, N-1$$

$$\mathbf{f}_l = \mathbf{0}, \quad \forall\, l \text{ in swing at step } k$$

$$\mathbf{t}_i^T \mathbf{f}_l - \mu\, \hat{\mathbf{n}}_l^T \mathbf{f}_l \leq 0, \quad i = 1, \ldots, 4$$

$$-\hat{\mathbf{n}}_l^T \mathbf{f}_l \leq -f_{\min}$$

The cost weights are $\mathbf{Q} = \text{diag}(1, 1, 50, 10, 20, 1, 2, 2, 1, 1, 1, 1)$ and $\mathbf{R} = 10^{-5}\mathbf{I}_{12}$. The large weight on $z$-position (50) and roll/pitch (10, 20) prioritizes body posture maintenance.

### D. Terrain-Adaptive Friction Cones

**This is a key integration point.** The friction cone constraints use surface normals $\hat{\mathbf{n}}_l$ queried from the global heightmap at each foot's predicted touchdown location. On flat ground, $\hat{\mathbf{n}} = [0, 0, 1]^T$ and the constraints reduce to the standard pyramid approximation $|f_x| \leq \mu f_z$, $|f_y| \leq \mu f_z$. On slopes, the normal is rotated, and we construct an orthonormal tangent basis $(\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2, \hat{\mathbf{n}})$ via Gram-Schmidt:

$$\hat{\mathbf{t}}_1 = \frac{\hat{\mathbf{n}} \times \mathbf{a}}{\|\hat{\mathbf{n}} \times \mathbf{a}\|}, \quad \hat{\mathbf{t}}_2 = \frac{\hat{\mathbf{n}} \times \hat{\mathbf{t}}_1}{\|\hat{\mathbf{n}} \times \hat{\mathbf{t}}_1\|}$$

where $\mathbf{a}$ is a reference vector not parallel to $\hat{\mathbf{n}}$. The four friction pyramid faces become:

$$(\hat{\mathbf{t}}_i - \mu\,\hat{\mathbf{n}})^T \mathbf{f}_l \leq 0, \quad (-\hat{\mathbf{t}}_i - \mu\,\hat{\mathbf{n}})^T \mathbf{f}_l \leq 0, \quad i = 1, 2$$

with $\mu = 0.8$ and minimum normal force $f_{\min} = 10$ N.

Without terrain-adaptive normals, the MPC assumes a flat contact surface and may command forces that violate the true friction cone on slopes, leading to foot slip. By feeding the heightmap normals directly into the QP constraints, the optimizer produces forces that respect the actual contact geometry.

### E. Solver

The QP is solved using OSQP with primal and dual warm-starting from the previous solution. The sparsity pattern of the Hessian $\mathbf{H}$ (block-diagonal) and constraint matrix $\mathbf{A}$ (banded dynamics + block-diagonal friction) is precomputed at initialization; only the numerical values of $\mathbf{B}_{d,k}$, $\mathbf{x}^{\text{ref}}$, and the friction matrix entries are updated each cycle. Typical solve times are 2--5 ms on a single CPU core.

---

## VII. REFERENCE TRAJECTORY GENERATION

The COM trajectory generator constructs the reference $\mathbf{x}^{\text{ref}}_{0:N-1}$ consumed by the MPC. It receives body-frame velocity commands $(v_x^{\text{cmd}}, v_y^{\text{cmd}}, \dot{\psi}^{\text{cmd}})$ from the MPPI planner.

### A. Position and Velocity

The world-frame reference velocity is $\mathbf{v}_{\text{ref}} = \mathbf{R}_z(\psi)\,[v_x^{\text{cmd}}, v_y^{\text{cmd}}, 0]^T$. Position is extrapolated linearly:

$$\mathbf{p}_k^{\text{ref}} = \mathbf{p}_{\text{current}} + \mathbf{v}_{\text{ref}}\,k\Delta t$$

with a position error clamp: $\|\mathbf{p}^{\text{ref}} - \mathbf{p}_{\text{current}}\|_\infty \leq 0.1$ m to prevent reference jumps.

### B. Terrain-Consistent Height

The reference height adapts to the terrain surface by querying the heightmap at each horizon step:

$$z_k^{\text{ref}} = \max\!\left(h^{\text{ground}}(x_k, y_k),\; \bar{h}_{\text{hip},k}\right) + z_{\text{des}}$$

where $\bar{h}_{\text{hip},k}$ is the mean ground height at the four predicted hip locations. This prevents body sag when the COM is above a depression but the feet land on elevated terrain.

### C. Slope-Aligned Orientation

The reference roll and pitch lightly track the terrain slope to keep the body roughly parallel to the local surface:

$$\phi_k^{\text{ref}} = -\kappa\,\arctan\!\left(\frac{n_y}{n_z}\right), \quad \theta_k^{\text{ref}} = \kappa\,\arctan\!\left(\frac{n_x}{n_z}\right)$$

with blending factor $\kappa = 0.25$. Full alignment ($\kappa = 1$) causes overshoot on rough terrain; light blending provides a soft reference that the MPC tracks without instability.

### D. Lever Arms and Contact Normals

At each swing-to-stance transition in the horizon, the trajectory generator computes the predicted touchdown position (Section VIII) and queries the heightmap for the surface normal $\hat{\mathbf{n}}_l$. These normals are stored in a $(4 \times N \times 3)$ array and passed to the MPC for friction cone construction (Section VI-D). The lever arms $\mathbf{r}_l$ are recomputed at each horizon step during stance as $\mathbf{r}_l = \mathbf{p}_l^{\text{td}} - \mathbf{p}_k^{\text{ref}}$.

---

## VIII. GAIT SCHEDULING AND FOOTHOLD SELECTION

### A. Trot Gait

We use a trotting gait with frequency $f_{\text{gait}} = 3$ Hz and duty cycle $D = 0.8$ (80% stance). The contact phase for leg $l$ at time $t$ is:

$$c_l(t) = \begin{cases} 1 & \text{if } \left(\frac{t}{T} + \phi_l\right) \bmod 1 < D \\ 0 & \text{otherwise} \end{cases}$$

with phase offsets $\boldsymbol{\phi} = [0, 0.25, 0.5, 0.75]$ producing the diagonal pairing (FL, RR) and (FR, RL).

### B. Touchdown Position

At the moment a leg lifts off, we compute the target touchdown position using a Raibert-style heuristic with multiple correction terms:

$$\mathbf{p}_{\text{td}} = \underbrace{\mathbf{p}_{\text{hip}}}_{\text{nominal}} + \underbrace{\mathbf{v}_{\text{des}}\,\tfrac{T_{\text{pred}}}{2}}_{\text{drift}} + \underbrace{k_p(\mathbf{p}_{\text{com}} - \mathbf{p}_{\text{des}})}_{\text{position}} + \underbrace{k_v(\dot{\mathbf{p}}_{\text{com}} - \mathbf{v}_{\text{des}})}_{\text{velocity}} + \underbrace{\boldsymbol{\delta}_{\text{yaw}}}_{\text{rotation}} + \underbrace{\boldsymbol{\delta}_{\text{capture}}}_{\text{roll-reactive}}$$

where $T_{\text{pred}} = t_{\text{swing}} + 0.5\,t_{\text{stance}}$, and the gains are $k_{v,x} = 0.4T$, $k_{p,x} = 0.1$, $k_{v,y} = 0.5T$, $k_{p,y} = 0.05$ (stronger lateral gains due to the Go2's narrow stance).

**Yaw rotation correction:**

$$\boldsymbol{\delta}_{\text{yaw}} = \begin{bmatrix} -\dot{\psi}\,T_{\text{pred}}\,r_y \\ \dot{\psi}\,T_{\text{pred}}\,r_x \\ 0 \end{bmatrix}$$

where $(r_x, r_y)$ is the hip offset from body center.

**Roll-reactive capture point (LIPM):** The lateral capture point is estimated using the Linear Inverted Pendulum Model:

$$\omega_0 = \sqrt{\frac{g}{h_{\text{com}}}}, \quad x_{\text{cap}} = \frac{v_{\text{lat}}}{\omega_0} - \sin(\phi)\,h_{\text{com}}$$

where $v_{\text{lat}}$ is lateral COM velocity in body frame and $\phi$ is roll angle. For a right leg (FR, RR), if $x_{\text{cap}} < 0$ (falling rightward), the correction widens the stance:

$$\boldsymbol{\delta}_{\text{capture}} = 0.6\,\mathbf{R}_z\,[0,\; x_{\text{cap}},\; 0]^T$$

Symmetric logic applies for left legs falling leftward.

### C. Terrain-Aware Foothold Selection

After computing the nominal touchdown $\mathbf{p}_{\text{td}}^0$, we refine it by searching 24 candidate locations (center + 8 inner ring at $0.5r$ + 12 outer ring at $r$ + 3 forward-biased, where $r = 0.15$ m). Candidates are rotated to align with the walking direction.

**Hard gates** (any failure eliminates the candidate):
- Surface slope $> 25^\circ$
- Hip-to-foothold distance $\notin [0.08, 0.35]$ m
- Height jump $|z - z_0| > 0.10$ m
- Plane-fit residual $> 0.015$ m (non-planar surface / holes)

**Support polygon stability.** For each surviving candidate, we compute the support polygon at touchdown time. The predicted COM position accounts for roll-induced lateral acceleration:

$$\mathbf{p}_{\text{com}}^{\text{pred}} = \mathbf{p}_{\text{com}} + (\dot{\mathbf{p}}_{\text{com}} + \dot{\mathbf{p}}_{\text{lat}})\,t_{\text{swing}} + \tfrac{1}{2}\,g\sin(\phi)\,t_{\text{swing}}^2\,\hat{\mathbf{y}}_{\text{world}}$$

The COM is projected onto the support plane (defined by the average surface normal of stance feet), and the signed distance to the nearest convex hull edge is computed. Candidates with margin $< 0$ (COM outside polygon) are rejected.

**Adaptive multi-objective scoring.** Surviving candidates are scored:

$$S = w_d\,\bar{d} + w_s\,\bar{s} + w_r\,\bar{r} + w_f\,\bar{f} + w_g\,\bar{g} + w_{\text{stab}}(\phi)\,\bar{m}$$

where $\bar{(\cdot)}$ denotes min-max normalization to $[0, 1]$, and:
- $d$ = distance from nominal touchdown
- $s$ = surface slope
- $r$ = plane-fit residual (roughness)
- $f$ = forward progress along walking direction
- $g$ = grade preference (front legs prefer uphill, rear prefer downhill)
- $m$ = stability margin deficit: $\max(0, m_{\text{target}} - m_{\text{actual}})$ with $m_{\text{target}} = 0.06$ m

The stability weight adapts to roll disturbance:

$$w_{\text{stab}}(\phi) = 0.20 + 0.50\,\text{clip}\!\left(\frac{|\phi|}{10^\circ} + \frac{|\dot{\phi}|}{1.0},\; 0,\; 1\right)$$

This increases the stability priority from 0.20 to 0.70 when the robot is actively rolling, producing wider stances that arrest the fall.

### D. Swing Trajectory

The swing foot follows a minimum-jerk trajectory between liftoff position $\mathbf{p}_0$ and touchdown $\mathbf{p}_f$:

$$\mathbf{p}(s) = \mathbf{p}_0 + (\mathbf{p}_f - \mathbf{p}_0)\,\mu(s), \quad \mu(s) = 10s^3 - 15s^4 + 6s^5$$

where $s = t / t_{\text{swing}} \in [0, 1]$. A smooth height bump provides ground clearance:

$$\Delta z(s) = h_{\text{sw}} \cdot 64\,s^3(1 - s)^3$$

where $h_{\text{sw}} = 0.15$ m. Both $\mu$ and $\Delta z$ have zero first and second derivatives at $s \in \{0, 1\}$, ensuring smooth liftoff and touchdown.

---

## IX. LOW-LEVEL LEG CONTROLLER

The leg controller runs at 200 Hz and maps MPC forces and swing trajectories to joint torques.

### A. Swing Phase

During swing, each leg tracks its trajectory (Section VIII-D) using Cartesian impedance control with inverse-dynamics feedforward:

$$\mathbf{f}_{\text{swing}} = \mathbf{K}_p(\mathbf{p}_{\text{des}} - \mathbf{p}) + \mathbf{K}_d(\dot{\mathbf{p}}_{\text{des}} - \dot{\mathbf{p}}) + \boldsymbol{\Lambda}(\ddot{\mathbf{p}}_{\text{des}} - \dot{\mathbf{J}}\dot{\mathbf{q}})$$

where $\boldsymbol{\Lambda} = (\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^T)^{-1}$ is the Cartesian inertia matrix, $\mathbf{J}$ is the $3 \times 18$ foot Jacobian, $\mathbf{M}$ is the $18 \times 18$ joint-space mass matrix, and $\dot{\mathbf{J}}\dot{\mathbf{q}}$ is the Jacobian time-derivative term. Gains are $\mathbf{K}_p = 400\,\mathbf{I}_3$ N/m and $\mathbf{K}_d = 75\,\mathbf{I}_3$ Ns/m.

### B. Stance Phase

During stance, the controller tracks the MPC contact force with a foothold-hold feedback:

$$\mathbf{f}_{\text{stance}} = -\mathbf{f}_{\text{mpc}} + \mathbf{K}_p^{\text{st}}(\mathbf{p}_{\text{td}} - \mathbf{p}) + \mathbf{K}_d^{\text{st}}(-\dot{\mathbf{p}})$$

with $\mathbf{K}_p^{\text{st}} = \text{diag}(300, 300, 0)$ N/m (no vertical position gain; the ground provides vertical constraint) and $\mathbf{K}_d^{\text{st}} = \text{diag}(40, 40, 20)$ Ns/m.

### C. Joint Torques

In both phases, the Cartesian force is mapped to joint torques with Coriolis and gravity compensation:

$$\boldsymbol{\tau} = \mathbf{J}_{\text{leg}}^T\,\mathbf{f} + (\mathbf{C}\dot{\mathbf{q}} + \mathbf{g})_{\text{leg}}$$

where $\mathbf{J}_{\text{leg}} \in \mathbb{R}^{3 \times 3}$ is the position Jacobian for the three leg joints, and $\mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$ are the Coriolis and gravitational torque terms from the Pinocchio rigid-body dynamics engine. Joint torques are saturated at 90% of hardware limits: $\tau_{\text{hip}} = 23.7$ Nm, $\tau_{\text{thigh}} = 23.7$ Nm, $\tau_{\text{knee}} = 45.4$ Nm.

---

## X. INTEGRATION: TERRAIN DATA FLOW

The central claim of this work is that tight coupling between the terrain representation and all control layers yields more robust locomotion than treating navigation and locomotion as independent modules. The following summarizes how the dual-layer heightmap $\mathcal{H}$ is consumed:

| Consumer | Terrain Query | Effect |
|----------|--------------|--------|
| A* planner | Steppability $\sigma$ per grid cell | Routes around poor stepping surfaces |
| MPPI planner | Steppability $\sigma$ at predicted footholds | Penalizes trajectories where feet would land on rough/steep terrain |
| MPPI planner | Obstacle cost map $\mathcal{C}$ | Avoids collisions |
| MPPI planner | Terrain gradient $\nabla h$ | Penalizes steep slopes |
| COM trajectory | Ground height $h^{\text{ground}}$ and normal $\hat{\mathbf{n}}$ | Adapts COM height and body orientation to terrain |
| Centroidal MPC | Surface normal $\hat{\mathbf{n}}_l$ at each foot | Rotates friction cone constraints to match contact geometry |
| Foothold selector | Slope, roughness, plane-fit residual, height | Multi-criteria terrain quality evaluation for each candidate |
| Swing trajectory | Touchdown height $h^{\text{ground}}$ at target | Sets correct foot landing height |

This tight coupling means that, for example, when the heightmap detects a slope:
1. The A* planner may route around it (if steep enough),
2. The MPPI penalizes trajectories that place feet on its steepest regions,
3. The COM trajectory tilts the body to match the slope,
4. The MPC rotates its friction cones to respect the sloped contact surface,
5. The foothold selector picks the most stable foothold considering the slope's effect on the support polygon.

All five responses emerge from the same height data, updated at the same rate, with no communication overhead between modules.
