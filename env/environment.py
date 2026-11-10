import gymnasium as gym
from typing import Sequence
import numpy as np
import numpy.typing as npt
import sympy as sp


class GeodesicGliderEnv(gym.Env):
    """
    Gymnasium environment: a thrust-controlled glider chasing a moving target
    on a Riemannian surface embedded in R^3.

    The surface is a graph r(u, v) = (u, v, f(u, v)). The target moves freely along a geodesic,
    the glider moves along a geodesic plus a controlled thrust. The agent has to learn to
    reach the (non-static) target in the shortest possible time.

    Observation (dict):
        local_state:   [u1, v1, v1_e1, v1_e2,  u2, v2, v2_e1, v2_e2]
                       chart coordinates and velocities (in the orthonormal tangent basis);
                       1 - glider, 2 - target
        global_state:  [x1, y1, z1, vx1, vy1, vz1,  x2, y2, z2, vx2, vy2, vz2]
                       positions and velocities in R^3
        ego_features:  [d.e1, d.e2, |d|, dv.e1, dv.e2, time_left]
                       target offset d and relative velocity dv projected onto the glider's
                       tangent basis, distance to the target, remaining episode time

    Action:
        [a_1, a_2] in [-1, 1]^2 - thrust acceleration in the local orthonormal basis (e1, e2).

    The episode ends with success ("completed"), when leaving the allowed region ("too far"),
    or on timeout.

    Note: only manifolds of the form r(u, v) = (u, v, f(u, v)) are supported.
    """

    def __init__(self, r_manifold: sp.Matrix, cords: Sequence[sp.Symbol]):
        """
        Initialize spaces and state vectors, and precompute the manifold kinematics.

        r_manifold: SymPy Matrix representing the vector r(u, v) = (u, v, f(u, v))
        cords: sequence of SymPy symbols [u, v]
        """
        super().__init__()

        # State layout: [u, v, du/dt, dv/dt] (chart coordinates and coordinate velocities).
        # These are placeholders; real (random) values are set in reset().
        self.glider_state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.target_state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.previous_distance = 0

        # The observation is a dict (1 - glider, 2 - target):
        #   local_state:  [u1, v1, v1_e1, v1_e2,  u2, v2, v2_e1, v2_e2]
        #   global_state: [x1, y1, z1, v1_x, v1_y, v1_z,  x2, y2, z2, v2_x, v2_y, v2_z]
        #   ego_features: [d.e1, d.e2, |d|, dv.e1, dv.e2, time_left]
        self.observation_space = gym.spaces.Dict(
            {
                "local_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
                "global_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                "ego_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

        # The action is the thrust vector [a_1, a_2] in the orthonormal tangent basis.
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        self._compute_manifold_kinematics(r_manifold, cords)

    def _compute_manifold_kinematics(self, r_manifold: sp.Matrix, cords: Sequence[sp.Symbol]):
        """
        Compute the first fundamental form (metric), its inverse and the Christoffel symbols
        of the second kind, then compile the symbolic expressions into fast NumPy functions.

        For a graph surface r = (u, v, f) the formulas simplify to:
            g_ij       = delta_ij + f_i * f_j
            det(g)     = 1 + f_u^2 + f_v^2
            Gamma^k_ij = f_k * f_ij / det(g)
        """
        u, v = cords[0], cords[1]
        f = r_manifold[2]

        # Partial derivatives of the height function.
        f_u = sp.diff(f, u)
        f_v = sp.diff(f, v)
        f_uu = sp.diff(f_u, u)
        f_uv = sp.diff(f_u, v)
        f_vv = sp.diff(f_v, v)

        det = 1 + f_u**2 + f_v**2

        # Metric tensor and its inverse (closed form for a 2x2 matrix).
        g00 = 1 + f_u**2
        g01 = f_u * f_v
        g11 = 1 + f_v**2

        g = sp.Matrix([[g00, g01], [g01, g11]])
        g_inv = sp.Matrix([[g11, -g01], [-g01, g00]]) / det

        # Gamma[k][i][j] = Gamma^k_ij; index 0 is u, index 1 is v. Symmetric in (i, j).
        Gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]

        Gamma[0][0][0] = f_u * f_uu / det
        Gamma[0][0][1] = f_u * f_uv / det
        Gamma[0][1][0] = Gamma[0][0][1]
        Gamma[0][1][1] = f_u * f_vv / det

        Gamma[1][0][0] = f_v * f_uu / det
        Gamma[1][0][1] = f_v * f_uv / det
        Gamma[1][1][0] = Gamma[1][0][1]
        Gamma[1][1][1] = f_v * f_vv / det

        # Compile everything to numeric functions of (u, v).
        self.g = sp.lambdify(cords, g)
        self.g_inv = sp.lambdify(cords, g_inv)
        self.christoffel_funcs = sp.lambdify(cords, Gamma)
        self.cords = cords
        self.r = sp.lambdify(cords, r_manifold)
        # Coordinate tangent vectors r_u = dr/du and r_v = dr/dv.
        self.r_u = sp.lambdify(cords, sp.Matrix([1, 0, f_u]))
        self.r_v = sp.lambdify(cords, sp.Matrix([0, 1, f_v]))

    def _compute_derivatives(self, state, action):
        """
        Compute dS/dt for S = [u, v, du/dt, dv/dt] from the geodesic equation with thrust:

            d(u, v)/dt   = (du/dt, dv/dt)
            d2x^k/dt2    = a^k - Gamma^k_ij * dx^i/dt * dx^j/dt

        `action` is the thrust acceleration in the coordinate basis (zero for the target).
        """
        res = np.array(state)  # copy of the state (keeps its dtype)
        dudt = state[2]
        dvdt = state[3]
        Gamma = self.christoffel_funcs(state[0], state[1])

        # Position derivatives are the current velocities.
        res[0] = dudt
        res[1] = dvdt

        # Velocity derivatives: thrust minus the geodesic (curvature) term.
        # Mixed terms are counted twice thanks to Gamma^k_uv == Gamma^k_vu.
        res[2] = action[0] - (
            Gamma[0][0][0] * dudt * dudt
            + 2 * Gamma[0][0][1] * dudt * dvdt
            + Gamma[0][1][1] * dvdt * dvdt
        )
        res[3] = action[1] - (
            Gamma[1][0][0] * dudt * dudt
            + 2 * Gamma[1][0][1] * dudt * dvdt
            + Gamma[1][1][1] * dvdt * dvdt
        )
        return res

    def _rk4_step(self, state, action, dt):
        """
        One step of the classical 4th-order Runge-Kutta integrator for
        S = [u, v, du/dt, dv/dt] with the right-hand side from _compute_derivatives().
        """
        k1 = self._compute_derivatives(state, action)
        k2 = self._compute_derivatives(state + k1 * dt / 2, action)
        k3 = self._compute_derivatives(state + k2 * dt / 2, action)
        k4 = self._compute_derivatives(state + k3 * dt, action)
        S_next = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        return S_next

    def _change_basis_in_tangent_space(self, state, vector, reverse=False):
        """
        Convert a tangent vector between the orthonormal basis (e1, e2) and the
        coordinate basis (f1, f2) = (r_u, r_v).

        The thrust is expressed in an orthonormal basis, so equal actions mean equal physical
        acceleration regardless of metric distortion, but the dynamics need coordinate components.

        Gram-Schmidt orthonormalization of (f1, f2):
            e1    = f1 / |f1|
            e2    = (f2 - alpha * f1) / |f2 - alpha * f1|,   alpha = <f1, f2> / <f1, f1>

        reverse=False: a*e1 + b*e2  ->  x*f1 + y*f2   (orthonormal -> coordinate)
        reverse=True:  x*f1 + y*f2  ->  a*e1 + b*e2   (coordinate -> orthonormal)
        """
        u, v = state[0], state[1]
        metric = self.g(u, v)
        alpha = metric[0, 1] / metric[0, 0]
        # |f2 - alpha * f1|
        e2_norm = np.sqrt(metric[1, 1] - 2 * alpha * metric[0, 1] + alpha * alpha * metric[0, 0])

        if not reverse:
            a, b = vector[0], vector[1]
            y = b / e2_norm
            x = a / np.sqrt(metric[0, 0]) - b * alpha / e2_norm
            return np.array([x, y])
        else:
            x, y = vector[0], vector[1]
            a = (x + alpha * y) * np.sqrt(metric[0, 0])
            b = y * e2_norm
            return np.array([a, b])

    def step(self, action):
        """
        Execute one simulation step:
        1. Clip the action to the unit disk, scale it by the maximum thrust and convert it
           to coordinate acceleration.
        2. Integrate glider and target states with RK4 (dt = 0.01).
        3. Compute the new observation, termination flags and reward.

        Returns: (observation, reward, terminated, truncated, info)
        """
        # Never allow an action longer than 1 
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm

        
        action = action*self.maximum_thrust
        action = self._change_basis_in_tangent_space(self.glider_state, action)

        # The glider is thrust-controlled, the target moves freely (zero thrust).
        self.glider_state = self._rk4_step(self.glider_state, action, 0.01)
        self.target_state = self._rk4_step(self.target_state, np.array([0, 0]), 0.01)
        self.time += 0.01

        # Global state *before* the update: needed to check the segment travelled during this step.
        prev_global_state = self.cur_obs["global_state"]
        obs = self._get_observation()  # also refreshes self.cur_obs
        self._get_termination(prev_global_state[:3], prev_global_state[6:9])
        reward = self._get_reward()

        terminated = self.is_terminated
        truncated = self.is_truncated
        info = {
            "reason": self.reason,
            "distance": self.previous_distance,
        }
        return obs, reward, terminated, truncated, info

    def _global_velocity_vector(self, state):
        """
        Velocity vector V = (V_x, V_y, V_z) in R^3 for a chart velocity (du/dt, dv/dt):
            V = dr/dt = r_u * du/dt + r_v * dv/dt
        """
        u, v = state[0], state[1]
        dudt, dvdt = state[2], state[3]
        r_u_vec = np.array(self.r_u(u, v), dtype=np.float32).reshape(-1)
        r_v_vec = np.array(self.r_v(u, v), dtype=np.float32).reshape(-1)

        V = r_u_vec * dudt + r_v_vec * dvdt
        return V

    def _calculate_distance(self, N):
        """
        Estimate the distance between glider and target along the surface.

        The straight segment between the two points is split into N points; each point is
        mapped back onto the manifold via r(u, v) (using its x, y as chart coordinates), and
        the Euclidean lengths between consecutive mapped points are summed. Cheap approximation
        of the geodesic distance.
        """
        ug, vg = self.glider_state[0], self.glider_state[1]
        ut, vt = self.target_state[0], self.target_state[1]
        g_pos = self.r(ug, vg)
        t_pos = self.r(ut, vt)

        # Step vector of the straight segment (N points -> N - 1 intervals).
        step_vec = (t_pos - g_pos) / (N - 1)

        points = np.zeros((N, 3))
        dist = 0
        for i in range(N):
            # Point on the straight segment, then lifted onto the surface.
            cur = g_pos + i * step_vec
            points[i] = cur.reshape(-1)
            cur = self.r(points[i][0], points[i][1])
            points[i] = cur.reshape(-1)
            if i > 0:
                dist += np.linalg.norm(points[i] - points[i - 1])
        return dist

    def _get_termination(self, prev_g, prev_t):
        """
        Update termination flags and `reason` for the current step.
        prev_g, prev_t: positions of glider and target in R^3 at the previous step.

        Success is checked on the whole segment travelled during the step (not only at its end),
        so a fast glider cannot "tunnel" through the target. For this we take the relative
        position p(s) = p0 + s * (p1 - p0), s in [0, 1], and find its closest point to the origin.
        """
        obs = self.cur_obs
        global_state = obs["global_state"]
        g = global_state[:3]
        t = global_state[6:9]

        # Relative position (glider - target) at the start and at the end of the step.
        p0, p1 = prev_g - prev_t, g - t
        seg = p1 - p0
        # Parameter of the closest approach, clipped to the segment.
        s = np.clip(-(p0 @ seg) / max(seg @ seg, 1e-12), 0.0, 1.0)

        if np.linalg.norm(p0 + s * seg) < self.complete_region:
            self.is_terminated, self.reason = True, "completed"
        elif max(np.max(np.abs(g)), np.max(np.abs(t))) > self.cord_limit:
            self.is_terminated, self.reason = True, "too far"
        elif self.time >= self.timeout:
            self.is_truncated, self.reason = True, "timeout"

    def _get_observation(self):
        """
        Build the observation dict for the policy:
        - local_state:  chart coordinates, velocities converted to the orthonormal basis (e1, e2)
        - global_state: positions and velocities in R^3
        - ego_features: target relative to the glider (see _ego_features)
        """
        glider = self.glider_state
        target = self.target_state

        # Velocities in R^3 (computed from the coordinate-basis states).
        V_g = self._global_velocity_vector(glider)
        V_t = self._global_velocity_vector(target)

        # Velocity components: coordinate basis -> orthonormal basis.
        g_v = self._change_basis_in_tangent_space(glider, glider[2:4], True)
        t_v = self._change_basis_in_tangent_space(target, target[2:4], True)

        # Local states [u, v, v_e1, v_e2]. (Names are reused: from here on they are no longer
        # the raw coordinate states.)
        glider = np.array([glider[0], glider[1], g_v[0], g_v[1]])
        target = np.array([target[0], target[1], t_v[0], t_v[1]])

        # Positions on the surface in R^3.
        g_pos = np.array(self.r(glider[0], glider[1])).reshape(-1)
        t_pos = np.array(self.r(target[0], target[1])).reshape(-1)

        obs = {
            "local_state": np.concatenate([glider, target]).astype(np.float32),
            "global_state": np.concatenate([g_pos, V_g, t_pos, V_t]).astype(np.float32),
        }
        # _ego_features() reads global_state from self.cur_obs, so it must be set first.
        self.cur_obs = obs
        obs["ego_features"] = self._ego_features()
        self.cur_obs = obs
        return obs

    def _get_reward(self):
        """
        Reward = progress towards the target - time penalty (+ bonus for success):
        - positive when the distance decreases, negative when it grows
        - the time penalty grows linearly during the episode
        - +5 for reaching the target
        """
        distance = self._calculate_distance(self.N)
        delta = self.distance_reward_scale * (self.previous_distance - distance)

        self.time_penalty += 0.0001
        self.previous_distance = distance

        if self.reason == "completed":
            delta += 5
        if self.reason == "too far":
            delta -= 0  # no extra penalty for now
        if self.reason == "timeout":
            delta -= 0  # no extra penalty for now
        return delta - self.time_penalty

    def reset(self, seed=None, options=None):
        """
        Reset the environment: randomly generate initial positions and velocities
        for glider and target (resampled until the episode does not end immediately).

        Default options:
        options = {
            'timeout': 10,
            'cord_limit': 5.0,
            'maximum_thrust': 10,
            'complete_region': 1e-2,
            'distance_reward_scale': 2,
            'mu_gl': 0,
            'sigma_gl': cord_limit/2,
            'mu_tg': 0,
            'sigma_tg': cord_limit/2,
            'mu_gl_vel': 0,
            'sigma_gl_vel': 0.3,
            'mu_tg_vel': 0,
            'sigma_tg_vel': 0.3,
            'N': 10
        }

        Positions and velocities are drawn from normal distributions N(mu, sigma),
        the same mu/sigma for both chart coordinates.

        Returns: (observation, info)
        """
        # Force the first iteration of the resampling loop.
        self.is_terminated = True
        self.is_truncated = True
        rng = np.random.default_rng(seed)

        # An empty/None options dict means "use all defaults".
        opts = options or {}

        while self.is_terminated or self.is_truncated:
            self.is_terminated = False
            self.is_truncated = False
            self.time = 0
            self.time_penalty = 0.01
            self.reason = "None"

            # Episode and reward parameters.
            self.timeout = opts.get('timeout', 10)
            self.cord_limit = opts.get('cord_limit', 5.0)
            self.maximum_thrust = opts.get('maximum_thrust', 10)
            self.complete_region = opts.get('complete_region', 1e-2)
            self.distance_reward_scale = opts.get('distance_reward_scale', 2)

            # Initial-state distributions (position and velocity of glider / target).
            mu_gl = opts.get('mu_gl', 0)
            sigma_gl = opts.get('sigma_gl', self.cord_limit / 2)
            mu_tg = opts.get('mu_tg', 0)
            sigma_tg = opts.get('sigma_tg', self.cord_limit / 2)
            mu_gl_vel = opts.get('mu_gl_vel', 0)
            sigma_gl_vel = opts.get('sigma_gl_vel', 0.3)
            mu_tg_vel = opts.get('mu_tg_vel', 0)
            sigma_tg_vel = opts.get('sigma_tg_vel', 0.3)

            # Number of points for the distance estimate.
            N = opts.get('N', 10)
            self.N = N

            # State = [u, v, du/dt, dv/dt]. Sampling order matters for seeded reproducibility:
            # glider position, glider velocity, target position, target velocity.
            self.glider_state = np.concatenate(
                (rng.normal(mu_gl, sigma_gl, 2), rng.normal(mu_gl_vel, sigma_gl_vel, 2))
            )
            self.target_state = np.concatenate(
                (rng.normal(mu_tg, sigma_tg, 2), rng.normal(mu_tg_vel, sigma_tg_vel, 2))
            )
            self.previous_distance = self._calculate_distance(N)
            self.initial_distance = self.previous_distance

            self.rng = rng
            obs = self._get_observation()
            global_state = obs["global_state"]
            # Previous positions are taken equal to the current ones: rejects starts that are
            # already inside the success region or outside the allowed region.
            self._get_termination(global_state[:3], global_state[6:9])

        info = {
            "distance": self.previous_distance,
        }
        return obs, info

    def _ego_features(self):
        """
        Target position and velocity relative to the glider, expressed in the glider's
        tangent basis (e1, e2) built from r_u and r_v by Gram-Schmidt:

            [d.e1, d.e2, |d|, dv.e1, dv.e2, time_left]

        d  - vector from glider to target in R^3
        dv - target velocity minus glider velocity in R^3
        """
        g, t = self.glider_state, self.target_state
        global_state = self.cur_obs['global_state']
        gp, tp = global_state[0:3], global_state[6:9]

        # Orthonormal tangent basis at the glider's position.
        ru = np.asarray(self.r_u(g[0], g[1]), float).reshape(-1)
        rv = np.asarray(self.r_v(g[0], g[1]), float).reshape(-1)
        e1 = ru / np.linalg.norm(ru)
        e2 = rv - (rv @ e1) * e1
        e2 /= np.linalg.norm(e2)

        d = tp - gp
        dv = self._global_velocity_vector(t) - self._global_velocity_vector(g)
        return np.array(
            [d @ e1, d @ e2, np.linalg.norm(d), dv @ e1, dv @ e2, self.timeout - self.time],
            dtype=np.float32,
        )