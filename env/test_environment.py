"""
Tests for GeodesicGliderEnv (env/environment.py): geometry, dynamics and the RL interface.

Run from the project root:
    python -m pytest env/test_environment.py -v
or simply:
    python env/test_environment.py

The math is checked against *independent* references instead of re-implementing the
formulas of the environment:
    - tangent vectors and Christoffel symbols  <-> finite differences of the surface itself
      (plus hand-computed values for the paraboloid)
    - basis changes                            <-> an explicit Gram-Schmidt basis in R^3
    - the RK4 integrator                       <-> conservation laws of free geodesic motion
      (energy, Clairaut's integral on surfaces of revolution) and its 4th-order convergence
    - the distance estimate                    <-> the exact arc length of a parabola
    - observations                             <-> quantities recomputed from the raw state in R^3
    - the whole interface                      <-> an end-to-end check: a PD controller built only
      from the observations must reach a static target on every surface

Every geometry / dynamics test runs on several surfaces (see SURFACES), including the surface
used for training.

A few numbers below mirror constants hard-coded in the environment (time step, reward terms).
They are collected at the top of the file and act as the specification of the reward: if you
change them in the environment, change them here too.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import sympy as sp

# Make the project root (for `env.environment`) and this folder (flat layout) importable,
# so that the tests work with both `pytest` and `python -m pytest`.
_HERE = Path(__file__).resolve().parent
for _path in (_HERE.parent, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from env.environment import GeodesicGliderEnv
except ModuleNotFoundError:
    from environment import GeodesicGliderEnv


# ======================================================================================
# Specification constants 
# ======================================================================================
DT = 0.01                  # integration step and time increment per env.step()
TIME_PENALTY_START = 0.01  # time penalty right after reset()
TIME_PENALTY_STEP = 1e-4   # the penalty grows by this amount at every reward evaluation
SUCCESS_BONUS = 5.0        # extra reward on "completed"

U, V = sp.symbols("u v")

# Surfaces z = f(u, v) used in the tests.
SURFACES = {
    "flat": lambda u, v: sp.Integer(0),
    "tilted_plane": lambda u, v: 0.5 * u - 0.3 * v,        # constant metric, zero curvature
    "paraboloid": lambda u, v: u**2 + v**2,
    "saddle": lambda u, v: u * v,                           # only mixed second derivative
    "gaussian_bump": lambda u, v: sp.exp(-(u**2 + v**2) / 4),
    # Copy of the surface from agent/train.py (update here if you change it there).
    "training_surface": lambda u, v: (
        1.1 * sp.exp(-((u - 2.0)**2 + (v - 1.0)**2) / 2.0)
        - 1.2 * sp.exp(-((u + 2.0)**2 + (v + 2.0)**2) / 2.5)
        + 0.7 * sp.exp(-((u + 1.0)**2 + (v - 2.0)**2) / 4.0)
        + 0.25 * sp.sin(0.9 * u) * sp.cos(0.9 * v)
    ),
}
ZERO_CURVATURE_SURFACES = ("flat", "tilted_plane")
REVOLUTION_SURFACES = ("paraboloid", "gaussian_bump")  # symmetric about the z axis

# Points (u, v) at which the geometry is checked.
POINTS = [(0.3, -0.7), (1.2, 0.4), (-1.8, 1.5), (0.0, 0.0), (2.5, -2.0)]

# Options that keep episodes alive (nothing ends by itself) and starts well separated.
CALM_OPTIONS = dict(
    timeout=1000.0, cord_limit=1000.0, maximum_thrust=10.0, complete_region=0.05,
    distance_reward_scale=2.0,
    mu_gl=0.5, sigma_gl=0.5, mu_tg=-0.5, sigma_tg=0.5,
    mu_gl_vel=0.0, sigma_gl_vel=0.5, mu_tg_vel=0.0, sigma_tg_vel=0.5,
    N=10,
)


# ======================================================================================
# Helpers
# ======================================================================================
_ENV_CACHE = {}


def _get_env(name):
    """Build the environment for a surface once per session (SymPy -> NumPy compilation is slow)."""
    if name not in _ENV_CACHE:
        r_manifold = sp.Matrix([U, V, SURFACES[name](U, V)])
        _ENV_CACHE[name] = GeodesicGliderEnv(r_manifold, cords=[U, V])
    return _ENV_CACHE[name]


def reset_env(env, seed=0, **overrides):
    """reset() with CALM_OPTIONS (optionally overridden); returns (obs, info)."""
    return env.reset(seed=seed, options={**CALM_OPTIONS, **overrides})


def clear_flags(env):
    env.is_terminated = False
    env.is_truncated = False
    env.reason = "None"


def set_states(env, glider, target):
    """
    Put the environment into a chosen state [u, v, du/dt, dv/dt] for both bodies
    (what reset() does after sampling) and return the fresh observation.
    """
    env.glider_state = np.array(glider, dtype=float)
    env.target_state = np.array(target, dtype=float)
    env.previous_distance = env._calculate_distance(env.N)
    clear_flags(env)
    return env._get_observation()


def surface_point(env, u, v):
    """Point r(u, v) on the surface as a flat array of length 3."""
    return np.asarray(env.r(u, v), dtype=float).reshape(-1)


def tangent_vectors(env, u, v):
    """(r_u, r_v) as flat arrays of length 3."""
    return (np.asarray(env.r_u(u, v), dtype=float).reshape(-1),
            np.asarray(env.r_v(u, v), dtype=float).reshape(-1))


def orthonormal_basis(env, u, v):
    """Gram-Schmidt on (r_u, r_v) in R^3, computed independently of the environment code."""
    ru, rv = tangent_vectors(env, u, v)
    e1 = ru / np.linalg.norm(ru)
    e2 = rv - (rv @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return e1, e2


def velocity_in_r3(env, state):
    """Velocity r_u * du/dt + r_v * dv/dt in R^3 for a state [u, v, du/dt, dv/dt]."""
    ru, rv = tangent_vectors(env, state[0], state[1])
    return ru * state[2] + rv * state[3]


def speed_squared(env, state):
    """Riemannian squared speed g_ij v^i v^j of a state [u, v, du/dt, dv/dt]."""
    g = np.asarray(env.g(state[0], state[1]), dtype=float)
    velocity = np.asarray(state[2:4], dtype=float)
    return velocity @ g @ velocity


def free_flow(env, state, steps, dt=DT):
    """Integrate free (zero-thrust) motion with the environment's RK4 integrator."""
    state = np.array(state, dtype=float)
    zero_thrust = np.zeros(2)
    for _ in range(steps):
        state = env._rk4_step(state, zero_thrust, dt)
    return state


def evaluate_termination(env, prev_glider=None, prev_target=None):
    """
    Run the termination check and return (terminated, truncated, reason).
    By default the "previous" positions equal the current ones (no motion during the step).
    """
    global_state = env.cur_obs["global_state"]
    prev_glider = global_state[:3] if prev_glider is None else np.asarray(prev_glider, dtype=float)
    prev_target = global_state[6:9] if prev_target is None else np.asarray(prev_target, dtype=float)
    env._get_termination(prev_glider, prev_target)
    return env.is_terminated, env.is_truncated, env.reason


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture(params=list(SURFACES))
def env(request):
    """Every test using this fixture runs once per surface."""
    environment = _get_env(request.param)
    reset_env(environment)
    return environment


@pytest.fixture
def flat_env():
    environment = _get_env("flat")
    reset_env(environment)
    return environment


@pytest.fixture
def paraboloid_env():
    environment = _get_env("paraboloid")
    reset_env(environment)
    return environment


@pytest.fixture(params=ZERO_CURVATURE_SURFACES)
def zero_curvature_env(request):
    environment = _get_env(request.param)
    reset_env(environment)
    return environment


@pytest.fixture(params=REVOLUTION_SURFACES)
def revolution_env(request):
    environment = _get_env(request.param)
    reset_env(environment)
    return environment


# ======================================================================================
# Gymnasium interface
# ======================================================================================
def test_spaces_match_the_documented_interface(env):
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, [-1, -1])
    np.testing.assert_array_equal(env.action_space.high, [1, 1])

    assert isinstance(env.observation_space, gym.spaces.Dict)
    expected_shapes = {"local_state": (8,), "global_state": (12,), "ego_features": (6,)}
    assert set(env.observation_space.spaces) == set(expected_shapes)
    for key, shape in expected_shapes.items():
        space = env.observation_space.spaces[key]
        assert space.shape == shape, f"'{key}' shape"
        assert space.dtype == np.float32, f"'{key}' dtype"


def test_reset_and_step_follow_the_gymnasium_api(env):
    obs, info = reset_env(env, seed=42)
    assert isinstance(info, dict) and "distance" in info
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.5], dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and {"reason", "distance"} <= set(info)


def test_observations_are_finite_float32(env):
    obs, _ = reset_env(env, seed=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        for key, value in obs.items():
            assert value.dtype == np.float32, key
            assert np.all(np.isfinite(value)), key
        obs, *_ = env.step(rng.uniform(-1, 1, 2))



def test_step_does_not_modify_the_callers_action(flat_env):
    action = np.array([0.3, 0.4])
    original = action.copy()
    flat_env.step(action)
    np.testing.assert_array_equal(action, original)


# ======================================================================================
# Surface geometry: metric and Christoffel symbols
# ======================================================================================
@pytest.mark.parametrize("point", POINTS)
def test_tangent_vectors_match_finite_differences(env, point):
    u, v = point
    h = 1e-6
    ru_fd = (surface_point(env, u + h, v) - surface_point(env, u - h, v)) / (2 * h)
    rv_fd = (surface_point(env, u, v + h) - surface_point(env, u, v - h)) / (2 * h)
    ru, rv = tangent_vectors(env, u, v)
    np.testing.assert_allclose(ru, ru_fd, atol=1e-7)
    np.testing.assert_allclose(rv, rv_fd, atol=1e-7)


@pytest.mark.parametrize("point", POINTS)
def test_metric_is_the_gram_matrix_of_the_tangent_vectors(env, point):
    u, v = point
    ru, rv = tangent_vectors(env, u, v)
    gram = np.array([[ru @ ru, ru @ rv], [rv @ ru, rv @ rv]])
    np.testing.assert_allclose(env.g(u, v), gram, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("point", POINTS)
def test_inverse_metric_is_the_inverse(env, point):
    u, v = point
    product = np.asarray(env.g(u, v), dtype=float) @ np.asarray(env.g_inv(u, v), dtype=float)
    np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


def christoffel_from_metric_derivatives(env, u, v, h=1e-5):
    """
    Christoffel symbols of the second kind from the textbook formula
        Gamma^k_ij = 1/2 g^{kl} (d_i g_lj + d_j g_li - d_l g_ij),
    with the metric derivatives taken by central finite differences.
    Returns an array indexed [k, i, j].
    """
    point = np.array([u, v], dtype=float)

    def metric(p):
        return np.asarray(env.g(p[0], p[1]), dtype=float)

    dg = np.zeros((2, 2, 2))  # dg[i, l, j] = d g_lj / d x^i
    for i in range(2):
        step = np.zeros(2)
        step[i] = h
        dg[i] = (metric(point + step) - metric(point - step)) / (2 * h)

    g_inv = np.linalg.inv(metric(point))
    gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gamma[k, i, j] = 0.5 * sum(
                    g_inv[k, l] * (dg[i, l, j] + dg[j, l, i] - dg[l, i, j]) for l in range(2)
                )
    return gamma


@pytest.mark.parametrize("point", POINTS)
def test_christoffel_symbols_match_finite_differences(env, point):
    u, v = point
    expected = christoffel_from_metric_derivatives(env, u, v)
    actual = np.asarray(env.christoffel_funcs(u, v), dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("point", POINTS)
def test_christoffel_symbols_are_symmetric_in_the_lower_indices(env, point):
    gamma = np.asarray(env.christoffel_funcs(*point), dtype=float)
    np.testing.assert_array_equal(gamma, gamma.transpose(0, 2, 1))


def test_flat_plane_has_identity_metric_and_zero_christoffel(flat_env):
    np.testing.assert_allclose(flat_env.g(5.0, -3.0), np.eye(2))
    np.testing.assert_allclose(flat_env.g_inv(5.0, -3.0), np.eye(2))
    np.testing.assert_allclose(np.asarray(flat_env.christoffel_funcs(5.0, -3.0), dtype=float),
                               np.zeros((2, 2, 2)))


def test_tilted_plane_has_constant_metric_and_zero_christoffel():
    env = _get_env("tilted_plane")
    grad = np.array([0.5, -0.3])
    expected_metric = np.eye(2) + np.outer(grad, grad)  # g = I + grad(f) grad(f)^T
    for point in POINTS:
        np.testing.assert_allclose(env.g(*point), expected_metric, atol=1e-12)
        np.testing.assert_allclose(np.asarray(env.christoffel_funcs(*point), dtype=float),
                                   np.zeros((2, 2, 2)), atol=1e-12)


def test_paraboloid_christoffel_symbols_match_hand_computation(paraboloid_env):
    # z = u^2 + v^2 at (1, 0.5): f_u = 2, f_v = 1, f_uu = f_vv = 2, f_uv = 0, det = 1 + 4 + 1 = 6
    # Gamma^k_ij = f_k f_ij / det
    gamma = np.asarray(paraboloid_env.christoffel_funcs(1.0, 0.5), dtype=float)
    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 2 * 2 / 6
    expected[0, 1, 1] = 2 * 2 / 6
    expected[1, 0, 0] = 1 * 2 / 6
    expected[1, 1, 1] = 1 * 2 / 6
    np.testing.assert_allclose(gamma, expected, atol=1e-12)


# ======================================================================================
# Basis change: orthonormal (action / observation) <-> coordinate (dynamics)
# ======================================================================================
@pytest.mark.parametrize("point", POINTS)
def test_orthonormal_to_coordinate_conversion_preserves_the_norm(env, point):
    state = np.array([*point, 0.0, 0.0])
    g = np.asarray(env.g(*point), dtype=float)
    for action in ([1.0, 0.0], [0.0, 1.0], [0.6, -0.8], [0.3, 0.4], [-0.9, 0.2]):
        action = np.array(action)
        coordinate = env._change_basis_in_tangent_space(state, action)
        np.testing.assert_allclose(coordinate @ g @ coordinate, action @ action, rtol=1e-10)


@pytest.mark.parametrize("point", POINTS)
def test_basis_change_round_trip(env, point):
    state = np.array([*point, 0.0, 0.0])
    vector = np.array([0.37, -1.21])
    forward = env._change_basis_in_tangent_space(state, vector)
    np.testing.assert_allclose(env._change_basis_in_tangent_space(state, forward, reverse=True),
                               vector, atol=1e-12)
    backward = env._change_basis_in_tangent_space(state, vector, reverse=True)
    np.testing.assert_allclose(env._change_basis_in_tangent_space(state, backward),
                               vector, atol=1e-12)


@pytest.mark.parametrize("point", POINTS)
def test_basis_change_uses_the_gram_schmidt_basis_of_the_tangent_plane(env, point):
    # The components (a, b) in the orthonormal basis and (x, y) in the coordinate basis must
    # describe the same vector of R^3:  a*e1 + b*e2 == x*r_u + y*r_v.
    # (This also pins down the orientation of the basis, not only the lengths.)
    state = np.array([*point, 0.0, 0.0])
    a, b = 0.8, -0.6
    x, y = env._change_basis_in_tangent_space(state, np.array([a, b]))
    ru, rv = tangent_vectors(env, *point)
    e1, e2 = orthonormal_basis(env, *point)
    np.testing.assert_allclose(x * ru + y * rv, a * e1 + b * e2, atol=1e-12)


# ======================================================================================
# Dynamics: RK4 integration of the geodesic equation with thrust
# ======================================================================================
def test_body_at_rest_without_thrust_stays_at_rest(env):
    state = np.array([1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(env._rk4_step(state, np.zeros(2), DT), state, atol=1e-15)


def test_free_motion_on_a_zero_curvature_surface_is_a_straight_line(zero_curvature_env):
    # Gamma = 0  =>  u'' = v'' = 0  =>  coordinates change linearly in time.
    final = free_flow(zero_curvature_env, [0.0, 0.0, 2.0, 3.0], steps=100)
    np.testing.assert_allclose(final, [2.0, 3.0, 2.0, 3.0], atol=1e-12)


def test_constant_thrust_on_a_zero_curvature_surface_gives_uniform_acceleration(zero_curvature_env):
    # u(t) = u0 + v0 t + a t^2 / 2 (RK4 is exact for polynomials of this degree).
    env = zero_curvature_env
    acceleration = np.array([2.0, -1.0])  # coordinate acceleration
    state = np.array([1.0, -1.0, 0.5, 0.25])
    for _ in range(100):  # t = 1
        state = env._rk4_step(state, acceleration, DT)
    expected_position = np.array([1.0, -1.0]) + np.array([0.5, 0.25]) + 0.5 * acceleration
    expected_velocity = np.array([0.5, 0.25]) + acceleration
    np.testing.assert_allclose(state[:2], expected_position, atol=1e-10)
    np.testing.assert_allclose(state[2:], expected_velocity, atol=1e-10)


def test_free_geodesic_motion_conserves_speed(env):
    # Without thrust the Riemannian speed g(v, v) is a constant of motion (geodesics have
    # constant speed). RK4 with dt = 0.01 must keep it up to a tiny drift.
    state = np.array([0.4, -0.3, 0.8, 0.5])
    initial = speed_squared(env, state)
    for _ in range(100):  # T = 1
        state = env._rk4_step(state, np.zeros(2), DT)
        assert speed_squared(env, state) == pytest.approx(initial, rel=1e-5)


def test_free_motion_on_a_surface_of_revolution_conserves_angular_momentum(revolution_env):
    # For z = f(sqrt(u^2 + v^2)) the angle around the z axis is cyclic, so Clairaut's integral
    # L = u v' - v u' (= rho^2 * dphi/dt) is conserved along geodesics.
    env = revolution_env
    state = np.array([0.8, 0.3, -0.4, 0.9])
    initial = state[0] * state[3] - state[1] * state[2]
    for _ in range(150):
        state = env._rk4_step(state, np.zeros(2), DT)
        assert state[0] * state[3] - state[1] * state[2] == pytest.approx(initial, rel=1e-5)


def test_meridians_are_geodesics(revolution_env):
    # Starting on the u axis and moving along it, a free body stays on the u axis (by symmetry).
    final = free_flow(revolution_env, [0.7, 0.0, 0.5, 0.0], steps=200)
    assert abs(final[1]) < 1e-12 and abs(final[3]) < 1e-12
    assert final[0] > 0.7  # and it actually moved


def test_rk4_is_fourth_order_accurate(paraboloid_env):
    # Halving the step must reduce the error by about 2^4 = 16.
    state0 = [0.5, 0.2, 0.9, -0.6]
    total_time = 0.4

    def final_state(steps):
        return free_flow(paraboloid_env, state0, steps, dt=total_time / steps)

    reference = final_state(1280)
    error_coarse = np.linalg.norm(final_state(10) - reference)
    error_fine = np.linalg.norm(final_state(20) - reference)
    assert 12 < error_coarse / error_fine < 20


def test_step_advances_time_and_integrates_both_bodies(env):
    reset_env(env, seed=5)
    glider0, target0 = [0.4, -0.3, 0.8, 0.5], [-0.5, 0.6, -0.3, 0.2]
    set_states(env, glider0, target0)
    time_before = env.time

    env.step(np.zeros(2))

    zero = np.zeros(2)
    np.testing.assert_allclose(env.glider_state, env._rk4_step(np.array(glider0), zero, DT),
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(env.target_state, env._rk4_step(np.array(target0), zero, DT),
                               rtol=1e-12, atol=1e-14)
    assert env.time == pytest.approx(time_before + DT)


# ======================================================================================
# Action handling in step()
# ======================================================================================
@pytest.mark.parametrize("action", [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8), (0.3, 0.0)])
def test_thrust_produces_the_expected_physical_acceleration(env, action):
    # From rest, one step of thrust a (unit disk, orthonormal basis) must give the velocity
    # a * maximum_thrust * dt expressed in the orthonormal basis (up to the tiny curvature terms).
    reset_env(env, seed=0, maximum_thrust=10.0)
    set_states(env, glider=[0.3, -0.2, 0.0, 0.0], target=[-3.0, 3.0, 0.0, 0.0])
    obs, *_ = env.step(np.array(action, dtype=float))
    expected_velocity = 10.0 * DT * np.array(action)
    np.testing.assert_allclose(obs["local_state"][2:4], expected_velocity, atol=5e-4)


def test_actions_outside_the_unit_disk_are_clipped_to_it(env):
    reset_env(env, seed=2)
    glider0, target0 = env.glider_state.copy(), env.target_state.copy()

    env.step(np.array([3.0, 4.0]))  # norm 5
    clipped_result = env.glider_state.copy()

    set_states(env, glider0, target0)
    env.step(np.array([0.6, 0.8]))  # the same direction, norm 1
    np.testing.assert_allclose(clipped_result, env.glider_state, rtol=1e-12, atol=1e-14)


def test_actions_inside_the_unit_disk_are_not_normalised(flat_env):
    # Half of the maximum thrust must give half of the velocity change.
    reset_env(flat_env, seed=2, maximum_thrust=10.0)
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[-3.0, 3.0, 0.0, 0.0])
    flat_env.step(np.array([0.5, 0.0]))
    np.testing.assert_allclose(flat_env.glider_state[2:], [5.0 * DT, 0.0], atol=1e-12)


def test_target_is_not_thrust_controlled(env):
    reset_env(env, seed=2)
    set_states(env, glider=[0.3, 0.1, 0.0, 0.0], target=[-1.0, 1.0, 0.4, -0.2])
    target0 = env.target_state.copy()
    env.step(np.array([1.0, 0.0]))
    np.testing.assert_allclose(env.target_state, env._rk4_step(target0, np.zeros(2), DT),
                               rtol=1e-12, atol=1e-14)


# ======================================================================================
# Distance estimate
# ======================================================================================
def test_distance_on_a_flat_plane_is_euclidean(flat_env):
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])
    for n_points in (2, 5, 10, 50):
        assert flat_env._calculate_distance(n_points) == pytest.approx(5.0)


def test_distance_on_a_tilted_plane_is_exact():
    env = _get_env("tilted_plane")
    reset_env(env)
    set_states(env, glider=[1.0, -1.0, 0.0, 0.0], target=[3.0, 2.0, 0.0, 0.0])
    # chart displacement (2, 3), height difference 0.5 * 2 - 0.3 * 3 = 0.1
    expected = np.sqrt(2.0**2 + 3.0**2 + 0.1**2)
    for n_points in (2, 10, 50):
        assert env._calculate_distance(n_points) == pytest.approx(expected)


def test_distance_converges_to_the_arc_length_of_a_parabola(paraboloid_env):
    # Along the u axis the paraboloid is the parabola z = u^2; its length from 0 to 1 is
    # 1/2 sqrt(5) + asinh(2) / 4.
    exact = 0.5 * np.sqrt(5.0) + np.arcsinh(2.0) / 4.0
    set_states(paraboloid_env, glider=[0.0, 0.0, 0.0, 0.0], target=[1.0, 0.0, 0.0, 0.0])

    # Nested point sets: an inscribed polyline gets longer with refinement but never exceeds the arc.
    lengths = [paraboloid_env._calculate_distance(n) for n in (3, 5, 9, 17, 33, 65)]
    assert all(a < b for a, b in zip(lengths, lengths[1:]))
    assert all(length < exact for length in lengths)

    assert paraboloid_env._calculate_distance(4001) == pytest.approx(exact, abs=1e-6)
    # The default N = 10 is a cheap but decent approximation.
    assert paraboloid_env._calculate_distance(10) == pytest.approx(exact, rel=1e-2)


def test_distance_is_zero_for_coinciding_points_and_symmetric(env):
    set_states(env, glider=[0.4, -0.3, 0.0, 0.0], target=[0.4, -0.3, 0.0, 0.0])
    assert env._calculate_distance(10) == pytest.approx(0.0, abs=1e-12)

    set_states(env, glider=[0.4, -0.3, 0.0, 0.0], target=[-1.1, 0.9, 0.0, 0.0])
    forward = env._calculate_distance(10)
    set_states(env, glider=[-1.1, 0.9, 0.0, 0.0], target=[0.4, -0.3, 0.0, 0.0])
    assert env._calculate_distance(10) == pytest.approx(forward, rel=1e-12)


def test_distance_is_never_shorter_than_the_straight_chord(env):
    rng = np.random.default_rng(1)
    for _ in range(10):
        glider, target = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        set_states(env, glider=[*glider, 0.0, 0.0], target=[*target, 0.0, 0.0])
        chord = np.linalg.norm(surface_point(env, *glider) - surface_point(env, *target))
        assert env._calculate_distance(10) >= chord - 1e-12


# ======================================================================================
# Reward
# ======================================================================================
def test_reward_is_progress_towards_the_target_minus_the_time_penalty(flat_env):
    reset_env(flat_env, distance_reward_scale=2.5)
    set_states(flat_env, glider=[3.0, 3.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])  # distance 1
    flat_env.previous_distance = 5.0

    reward = flat_env._get_reward()

    assert reward == pytest.approx(2.5 * (5.0 - 1.0) - (TIME_PENALTY_START + TIME_PENALTY_STEP))
    assert reward > 0
    assert flat_env.previous_distance == pytest.approx(1.0)


def test_reward_is_negative_when_moving_away(flat_env):
    reset_env(flat_env, distance_reward_scale=2.5)
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])  # distance 5
    flat_env.previous_distance = 1.0

    assert flat_env._get_reward() == pytest.approx(
        2.5 * (1.0 - 5.0) - (TIME_PENALTY_START + TIME_PENALTY_STEP))


def test_time_penalty_grows_linearly(flat_env):
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])  # no progress
    for k in range(1, 6):
        assert flat_env._get_reward() == pytest.approx(-(TIME_PENALTY_START + k * TIME_PENALTY_STEP))


def test_success_bonus_is_added_on_completion(flat_env):
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])
    flat_env.reason = "completed"
    assert flat_env._get_reward() == pytest.approx(
        SUCCESS_BONUS - (TIME_PENALTY_START + TIME_PENALTY_STEP))


@pytest.mark.parametrize("reason", ["too far", "timeout"])
def test_failures_have_no_extra_penalty_for_now(flat_env, reason):
    # Current specification (see the comments in _get_reward): only the progress and time terms.
    set_states(flat_env, glider=[0.0, 0.0, 0.0, 0.0], target=[3.0, 4.0, 0.0, 0.0])
    flat_env.reason = reason
    assert flat_env._get_reward() == pytest.approx(-(TIME_PENALTY_START + TIME_PENALTY_STEP))


def test_episode_return_telescopes_to_the_total_progress(env):
    # The progress term is potential-based: the sum over an episode only depends on the first and
    # the last distance. Everything else is the time penalty and the success bonus.
    scale = 3.0
    _, info = reset_env(env, seed=7, distance_reward_scale=scale)
    initial_distance = info["distance"]
    rng = np.random.default_rng(0)

    total, steps = 0.0, 0
    for _ in range(40):
        _, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 2))
        total += reward
        steps += 1
        if terminated or truncated:
            break

    bonus = SUCCESS_BONUS if info["reason"] == "completed" else 0.0
    penalties = sum(TIME_PENALTY_START + TIME_PENALTY_STEP * k for k in range(1, steps + 1))
    assert total == pytest.approx(scale * (initial_distance - info["distance"]) + bonus - penalties)


# ======================================================================================
# Episode termination
# ======================================================================================
def test_success_is_detected_on_the_segment_travelled_during_the_step(flat_env):
    # The glider flies straight through the target between two steps: both end points are 0.5
    # away from it, but the swept segment passes through it ("no tunnelling").
    reset_env(flat_env, complete_region=0.1)
    set_states(flat_env, glider=[0.5, 0.0, 0.0, 0.0], target=[0.0, 0.0, 0.0, 0.0])
    terminated, truncated, reason = evaluate_termination(
        flat_env, prev_glider=[-0.5, 0.0, 0.0], prev_target=[0.0, 0.0, 0.0])
    assert terminated and not truncated and reason == "completed"


@pytest.mark.parametrize("offset, expected", [(0.05, True), (0.15, False)])
def test_success_region_is_measured_at_the_closest_approach(flat_env, offset, expected):
    reset_env(flat_env, complete_region=0.1)
    set_states(flat_env, glider=[0.5, offset, 0.0, 0.0], target=[0.0, 0.0, 0.0, 0.0])
    terminated, _, _ = evaluate_termination(
        flat_env, prev_glider=[-0.5, offset, 0.0], prev_target=[0.0, 0.0, 0.0])
    assert terminated is expected


def test_no_termination_and_no_nan_when_nothing_moves(flat_env):
    # The segment has zero length here: the closest-approach computation must not divide by zero.
    reset_env(flat_env, complete_region=0.1)
    set_states(flat_env, glider=[2.0, 0.0, 0.0, 0.0], target=[0.0, 0.0, 0.0, 0.0])
    with np.errstate(invalid="raise", divide="raise"):
        terminated, truncated, reason = evaluate_termination(flat_env)
    assert not terminated and not truncated and reason == "None"


@pytest.mark.parametrize("glider, target", [
    ([6.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),    # glider leaves the region
    ([0.0, 0.0, 0.0, 0.0], [0.0, -6.0, 0.0, 0.0]),   # target leaves the region
])
def test_leaving_the_allowed_region_terminates_the_episode(flat_env, glider, target):
    reset_env(flat_env, cord_limit=5.0)
    set_states(flat_env, glider=glider, target=target)
    terminated, truncated, reason = evaluate_termination(flat_env)
    assert terminated and not truncated and reason == "too far"


def test_staying_inside_the_allowed_region_does_not_terminate(flat_env):
    reset_env(flat_env, cord_limit=5.0)
    set_states(flat_env, glider=[4.9, -4.9, 0.0, 0.0], target=[0.0, 0.0, 0.0, 0.0])
    terminated, truncated, _ = evaluate_termination(flat_env)
    assert not terminated and not truncated


def test_timeout_truncates_the_episode(flat_env):
    reset_env(flat_env, timeout=0.5)
    flat_env.time = 0.5 - DT / 2
    _, _, terminated, truncated, info = flat_env.step(np.zeros(2))
    assert truncated and not terminated
    assert info["reason"] == "timeout"


# ======================================================================================
# reset()
# ======================================================================================
def test_reset_is_reproducible_for_a_given_seed(env):
    reset_env(env, seed=123)
    glider, target = env.glider_state.copy(), env.target_state.copy()
    reset_env(env, seed=123)
    np.testing.assert_array_equal(env.glider_state, glider)
    np.testing.assert_array_equal(env.target_state, target)

    reset_env(env, seed=124)
    assert not np.array_equal(env.glider_state, glider)


def test_reset_applies_options(flat_env):
    options = dict(
        timeout=3.0, cord_limit=50.0, maximum_thrust=7.0, complete_region=0.2,
        distance_reward_scale=4.0, N=25,
        mu_gl=1.0, sigma_gl=1e-3, mu_gl_vel=0.5, sigma_gl_vel=1e-3,
        mu_tg=-1.0, sigma_tg=1e-3, mu_tg_vel=-0.5, sigma_tg_vel=1e-3,
    )
    backup = dict(options)
    obs, info = flat_env.reset(seed=1, options=options)

    assert options == backup  # reset() must not modify the caller's dict
    assert (flat_env.timeout, flat_env.cord_limit, flat_env.maximum_thrust) == (3.0, 50.0, 7.0)
    assert (flat_env.complete_region, flat_env.distance_reward_scale, flat_env.N) == (0.2, 4.0, 25)
    np.testing.assert_allclose(flat_env.glider_state, [1.0, 1.0, 0.5, 0.5], atol=1e-2)
    np.testing.assert_allclose(flat_env.target_state, [-1.0, -1.0, -0.5, -0.5], atol=1e-2)


@pytest.mark.parametrize("options", [None, {}])
def test_reset_defaults(flat_env, options):
    flat_env.reset(seed=1, options=options)
    assert flat_env.timeout == 10
    assert flat_env.cord_limit == 5.0
    assert flat_env.maximum_thrust == 10
    assert flat_env.complete_region == pytest.approx(1e-2)
    assert flat_env.distance_reward_scale == 2
    assert flat_env.N == 10


@pytest.mark.parametrize("cord_limit", [10.0, 1000.0])
def test_default_position_spread_is_half_the_cord_limit(flat_env, cord_limit):
    # sigma defaults to cord_limit / 2: the median |u| of a N(0, sigma) sample is about 0.67 sigma
    # (a bit less here, because starts outside of the allowed region are rejected).
    sigma = cord_limit / 2
    positions = []
    for seed in range(300):
        flat_env.reset(seed=seed, options={"cord_limit": cord_limit})
        positions.append(abs(flat_env.glider_state[0]))
    assert 0.5 * sigma < np.median(positions) < 0.8 * sigma


def test_reset_clears_the_episode_state(env):
    reset_env(env, seed=1)
    for _ in range(5):
        env.step(np.zeros(2))
    assert env.time > 0 and env.time_penalty > TIME_PENALTY_START

    reset_env(env, seed=2)
    assert env.time == 0
    assert env.time_penalty == TIME_PENALTY_START
    assert (env.is_terminated, env.is_truncated, env.reason) == (False, False, "None")


def test_reset_reports_the_initial_distance(env):
    _, info = reset_env(env, seed=4)
    assert info["distance"] == pytest.approx(env._calculate_distance(env.N))
    assert env.previous_distance == pytest.approx(info["distance"])


def test_reset_never_returns_an_already_finished_episode(env):
    # With a big success region many random starts are rejected and resampled.
    for seed in range(30):
        obs, _ = reset_env(env, seed=seed, complete_region=0.5, cord_limit=3.0)
        assert (env.is_terminated, env.is_truncated, env.reason) == (False, False, "None")
        positions = obs["global_state"]
        assert np.linalg.norm(positions[:3] - positions[6:9]) >= 0.5
        assert np.max(np.abs(positions[[0, 1, 2, 6, 7, 8]])) <= 3.0


# ======================================================================================
# Observations
# ======================================================================================
GLIDER_STATE = [0.9, -0.4, 1.3, -0.7]
TARGET_STATE = [-1.1, 0.6, -0.5, 0.8]


@pytest.mark.parametrize("who, state, local_slice, global_slice", [
    ("glider", GLIDER_STATE, slice(0, 4), slice(0, 6)),
    ("target", TARGET_STATE, slice(4, 8), slice(6, 12)),
])
def test_local_and_global_states_match_the_raw_state(env, who, state, local_slice, global_slice):
    set_states(env, GLIDER_STATE, TARGET_STATE)
    obs = env._get_observation()
    state = np.array(state)
    u, v = state[:2]
    e1, e2 = orthonormal_basis(env, u, v)
    velocity = velocity_in_r3(env, state)

    local = obs["local_state"][local_slice]
    global_ = obs["global_state"][global_slice]

    # local_state: chart coordinates + velocity components along (e1, e2)
    np.testing.assert_allclose(local[:2], [u, v], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(local[2:], [velocity @ e1, velocity @ e2], rtol=1e-5, atol=1e-5)
    # global_state: position and velocity in R^3
    np.testing.assert_allclose(global_[:3], surface_point(env, u, v), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(global_[3:], velocity, rtol=1e-5, atol=1e-5)
    # the speed is the same in all three descriptions
    speed = np.sqrt(speed_squared(env, state))
    assert np.linalg.norm(local[2:]) == pytest.approx(speed, rel=1e-5)
    assert np.linalg.norm(global_[3:]) == pytest.approx(speed, rel=1e-5)


def test_global_velocity_is_the_time_derivative_of_the_position(env):
    # V = dr/dt: compare with a finite difference of r along the chart velocity.
    state = np.array([0.9, -0.4, 1.3, -0.7])
    h = 1e-6
    ahead = surface_point(env, state[0] + h * state[2], state[1] + h * state[3])
    behind = surface_point(env, state[0] - h * state[2], state[1] - h * state[3])
    np.testing.assert_allclose(env._global_velocity_vector(state), (ahead - behind) / (2 * h),
                               rtol=1e-5, atol=1e-5)


def test_global_velocity_vector_shape_and_flat_plane_value(flat_env):
    state = np.array([1.0, 1.0, 2.0, 3.0])
    velocity = flat_env._global_velocity_vector(state)
    assert isinstance(velocity, np.ndarray) and velocity.shape == (3,)
    np.testing.assert_allclose(velocity, [2.0, 3.0, 0.0])


def test_ego_features_describe_the_target_relative_to_the_glider(env):
    reset_env(env, seed=1, timeout=7.0)
    set_states(env, GLIDER_STATE, TARGET_STATE)
    ego = env._get_observation()["ego_features"]

    e1, e2 = orthonormal_basis(env, *GLIDER_STATE[:2])
    offset = surface_point(env, *TARGET_STATE[:2]) - surface_point(env, *GLIDER_STATE[:2])
    relative_velocity = (velocity_in_r3(env, np.array(TARGET_STATE))
                         - velocity_in_r3(env, np.array(GLIDER_STATE)))
    expected = [offset @ e1, offset @ e2, np.linalg.norm(offset),
                relative_velocity @ e1, relative_velocity @ e2, 7.0]
    np.testing.assert_allclose(ego, expected, rtol=1e-5, atol=1e-5)


def test_ego_features_on_a_flat_plane_are_plain_differences(flat_env):
    reset_env(flat_env, seed=1, timeout=7.0)
    set_states(flat_env, GLIDER_STATE, TARGET_STATE)
    ego = flat_env._get_observation()["ego_features"]
    dx, dy = TARGET_STATE[0] - GLIDER_STATE[0], TARGET_STATE[1] - GLIDER_STATE[1]
    dvx, dvy = TARGET_STATE[2] - GLIDER_STATE[2], TARGET_STATE[3] - GLIDER_STATE[3]
    np.testing.assert_allclose(ego, [dx, dy, np.hypot(dx, dy), dvx, dvy, 7.0], rtol=1e-5)


def test_remaining_time_counts_down(env):
    reset_env(env, seed=1, timeout=7.0)
    obs, *_ = env.step(np.zeros(2))
    assert obs["ego_features"][5] == pytest.approx(7.0 - DT, abs=1e-5)
    obs, *_ = env.step(np.zeros(2))
    assert obs["ego_features"][5] == pytest.approx(7.0 - 2 * DT, abs=1e-5)


def test_info_reports_the_current_distance_and_reason(env):
    reset_env(env, seed=1)
    _, _, _, _, info = env.step(np.array([0.2, 0.1]))
    assert info["distance"] == pytest.approx(env._calculate_distance(env.N))
    assert info["reason"] == "None"


# ======================================================================================
# End-to-end behaviour
# ======================================================================================
def test_random_episode_stays_finite_and_ends_with_a_valid_reason(env):
    rng = np.random.default_rng(0)
    reset_env(env, seed=1, timeout=1.0, cord_limit=50.0)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 2))
        assert np.isfinite(reward)
        assert all(np.all(np.isfinite(value)) for value in obs.values())
        if terminated or truncated:
            break
    else:
        pytest.fail("the episode did not end although the timeout is 1 s (100 steps)")
    assert info["reason"] in ("completed", "too far", "timeout")
    assert truncated == (info["reason"] == "timeout")
    assert terminated == (info["reason"] in ("completed", "too far"))


def test_pd_controller_built_from_observations_reaches_a_static_target(env):
    # Checks that the action basis, the velocity basis and the ego features are all consistent:
    # steering with  a = 2 * (target offset) - 2.5 * (own velocity)  expressed in the observed
    # orthonormal basis only works if all of them use the same orientation.
    # The two bodies start close to each other on purpose: this controller follows the straight
    # chord to the target projected onto the tangent plane, so on a steep slope it can stall
    # where the chord is perpendicular to the surface (a limitation of the controller, not of
    # the environment).
    reset_env(env, seed=1, timeout=10.0, cord_limit=50.0, complete_region=0.1, maximum_thrust=10.0)
    obs = set_states(env, glider=[-0.6, 0.4, 0.0, 0.0], target=[0.6, -0.4, 0.0, 0.0])

    info = {"reason": "None"}
    for _ in range(1000):
        offset = obs["ego_features"][:2].astype(float)
        velocity = obs["local_state"][2:4].astype(float)
        obs, _, terminated, truncated, info = env.step(2.0 * offset - 2.5 * velocity)
        if terminated or truncated:
            break
    assert info["reason"] == "completed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))