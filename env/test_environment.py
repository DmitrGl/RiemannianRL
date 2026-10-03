import gymnasium as gym
import numpy as np
import sympy as sp
import time

from environment import GeodesicGliderEnv

def create_paraboloid_env():
    """
    Creates an environment with a paraboloid manifold z = u^2 + v^2.
    Useful for testing non-trivial curvature and non-zero Christoffel symbols.
    """
    u, v = sp.symbols('u v')
    r_manifold = sp.Matrix([u, v, u**2 + v**2])
    return GeodesicGliderEnv(r_manifold=r_manifold, cords=[u, v])

def create_flat_env():
    """
    Creates an environment with a flat Euclidean plane z = 0.
    Useful for testing baseline physics: straight lines, zero curvature, exact Euclidean distance.
    """
    u, v = sp.symbols('u v')
    r_manifold = sp.Matrix([u, v, 0])
    return GeodesicGliderEnv(r_manifold=r_manifold, cords=[u, v])

def test_gym_api_compliance(env):
    """
    Validates strictly that the environment complies with standard Gymnasium API requirements.
    """
    print("  [~] Running API compliance tests...")
    
    # 1. Check Spaces
    assert isinstance(env.action_space, gym.spaces.Box), "action_space must be Box"
    assert env.action_space.dtype == np.float32, "action_space dtype must be float32"
    assert env.action_space.shape == (2,), "action_space shape must be (2,)"
    
    assert isinstance(env.observation_space, gym.spaces.Dict), "observation_space must be Dict"
    for key in ["glider", "target"]:
        assert key in env.observation_space.spaces, f"observation_space missing {key}"
        assert env.observation_space.spaces[key].shape == (4,), f"{key} shape must be (4,)"
    
    # 2. Check Reset returns
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict), "reset() must return a dictionary observation"
    assert isinstance(info, dict), "reset() must return an info dictionary"
    
    # 3. Check Step returns
    action = np.array([0.5, -0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    
    assert isinstance(next_obs, dict), "step() observation must be a dict"
    assert isinstance(reward, (float, np.floating)), "step() reward must be a float"
    assert isinstance(terminated, bool), "step() terminated flag must be a boolean"
    assert isinstance(truncated, bool), "step() truncated flag must be a boolean"
    assert isinstance(step_info, dict), "step() info must be a dict"
    
    print("  [OK] API compliance passed.")

def test_flat_manifold_kinematics(env):
    """
    On a flat manifold z=0, the metric tensor 'g' must be the Identity matrix,
    and all Christoffel symbols must be strictly zero.
    """
    print("  [~] Running flat manifold kinematics tests...")
    
    # Evaluate at an arbitrary point (u=5, v=-3)
    u_val, v_val = 5.0, -3.0
    
    # 1. Metric Tensor (First Fundamental Form)
    g_matrix = env.g(u_val, v_val)
    np.testing.assert_allclose(g_matrix, np.eye(2), err_msg="metric tensor on flat plane must be identity")
    
    # 2. Inverse Metric Tensor
    g_inv_matrix = env.g_inv(u_val, v_val)
    np.testing.assert_allclose(g_inv_matrix, np.eye(2), err_msg="inverse metric tensor on flat plane must be identity")
    
    # 3. Christoffel Symbols
    gamma = env.christoffel_funcs(u_val, v_val)
    np.testing.assert_allclose(gamma, np.zeros((2,2,2)), err_msg="Christoffel symbols on flat plane must be zero")
    
    print("  [OK] Flat manifold kinematics passed.")

def test_action_isometry(env):
    """
    Validates that the transformation from orthonormal basis (action space) 
    to coordinate basis preserves the norm of the thrust vector.
    ||action||^2_Euclidean == ||coordinate_accel||^2_g
    """
    print("  [~] Running action isometry (norm preservation) tests...")
    
    env.reset(seed=10)
    # Pick a random state on the paraboloid
    state = env.glider_state
    u, v = state[0], state[1]
    
    # Define an arbitrary thrust action
    action = np.array([0.8, -0.6]) # Euclidean norm is 1.0
    action_norm_squared = action[0]**2 + action[1]**2
    
    # Map to tangent space coordinate basis
    coord_accel = env._action_to_coordinate_acceleration(state, action)
    x, y = coord_accel[0], coord_accel[1]
    
    # Calculate length of mapped vector using the metric tensor g_{ij}
    g = env.g(u, v)
    mapped_norm_squared = g[0][0]*x**2 + 2*g[0][1]*x*y + g[1][1]*y**2
    
    np.testing.assert_allclose(
        action_norm_squared, 
        mapped_norm_squared, 
        err_msg="action vector norm is not preserved after basis transformation. Orthonormal basis is incorrect."
    )
    print("  [OK] Action isometry passed.")

def test_rk4_geodesic_stability(env):
    """
    Tests RK4 integration under zero thrust. 
    1. Zero velocity + Zero thrust = No movement.
    2. Non-zero velocity + Zero thrust on flat plane = Straight line (constant velocity).
    """
    print("  [~] Running RK4 stability and geodesic tests...")
    
    # Test 1: Static stability
    env.glider_state = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(zero_action)
    np.testing.assert_allclose(
        obs["glider"], 
        [1.0, 1.0, 0.0, 0.0], 
        err_msg="glider with 0 velocity and 0 thrust must not move"
    )

    # Test 2: Constant velocity on flat plane (geodesic in Euclidean space)
    flat_env = create_flat_env()
    flat_env.reset() 
    

    flat_env.glider_state = np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float32)
    flat_env.target_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    dt = 0.01 
    flat_env.step(zero_action)
    
    expected_u = 0.0 + 2.0 * dt
    expected_v = 0.0 + 3.0 * dt
    
    np.testing.assert_allclose(
        flat_env.glider_state[0:2], 
        [expected_u, expected_v], 
        err_msg="glider on flat plane with 0 thrust must move in straight line with constant velocity"
    )
    np.testing.assert_allclose(
        flat_env.glider_state[2:4], 
        [2.0, 3.0], 
        err_msg="glider on flat plane with 0 thrust must not change velocity"
    )
    
    print("  [OK] RK4 stability and geodesics passed.")

def test_distance_and_reward_logic(env):
    """
    Validates distance metric calculations against standard Euclidean distance,
    and ensures the reward function properly penalizes and rewards the agent.
    """
    print("  [~] Running distance and reward bounds tests...")
    
    # On a flat manifold, environment distance calculation should perfectly match 3D Euclidean distance
    flat_env = create_flat_env()
    flat_env.reset()
    flat_env.glider_state = np.array([0.0, 0.0, 0.0, 0.0])
    flat_env.target_state = np.array([3.0, 4.0, 0.0, 0.0])
    
    dist = flat_env._calculate_distance(N=10)
    np.testing.assert_allclose(dist, 5.0, err_msg="distance on flat plane (z=0) must equal 2D euclidean distance")
    
    # Test reward logic: getting closer
    flat_env.previous_distance = 5.0
    
    # Manually move glider closer to target (to [3.0, 3.0])
    flat_env.glider_state = np.array([3.0, 3.0, 0.0, 0.0])
    flat_env.target_state = np.array([3.0, 4.0, 0.0, 0.0])
    
    new_dist = flat_env._calculate_distance(N=10) # Should be 1.0
    expected_delta = 5.0 - 1.0 # previous - current
    expected_reward = expected_delta - 0.01
    
    # Bypass step() and call _get_reward directly to isolate logic
    flat_env.N = 10
    actual_reward = flat_env._get_reward()
    np.testing.assert_allclose(actual_reward, expected_reward, err_msg="reward function did not return expected delta minus time penalty")
    assert actual_reward > 0, "reward must be positive when making significant progress towards target"

    # Test reward logic: getting further away
    flat_env.previous_distance = 1.0
    flat_env.glider_state = np.array([0.0, 0.0, 0.0, 0.0]) # teleport back to origin
    flat_env.target_state = np.array([3.0, 4.0, 0.0, 0.0])
    
    actual_negative_reward = flat_env._get_reward()
    assert actual_negative_reward < 0, "reward must be strictly negative when moving away from target"

    print("  [OK] Distance and reward bounds passed.")

def test_reset_options_parsing(env):
    """
    Ensures that the environment correctly parses the options tuple passed to reset().
    options = (mu, sigma2, N)
    """
    print("  [~] Running reset options tests...")
    
    mu_test, sigma_test, N_test = 100.0, 0.001, 50
    obs, _ = env.reset(seed=777, options=(mu_test, sigma_test, N_test))
    
    # Glider and target states should be initialized close to 'mu' since sigma is tiny
    for val in obs["glider"]:
        assert 99.0 < val < 101.0, f"state initialized far from mu={mu_test} despite tiny variance"
        
    assert env.N == N_test, "environment did not update N parameter from options tuple"
    
    print("  [OK] Reset options passed.")

def main():
    print("="*60)
    print("INITIALIZING GEODESIC GLIDER TEST SUITE")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize environments
    print("\n[1/2] Compiling Paraboloid Environment (SymPy -> NumPy)...")
    paraboloid_env = create_paraboloid_env()
    
    print("\n[2/2] Compiling Flat Plane Environment (SymPy -> NumPy)...")
    flat_env = create_flat_env()
    
    print(f"\nEnvironments compiled in {time.time() - start_time:.2f} seconds. Starting tests...\n")
    
    # Run all test suites
    test_gym_api_compliance(paraboloid_env)
    test_flat_manifold_kinematics(flat_env)
    test_action_isometry(paraboloid_env)
    test_rk4_geodesic_stability(paraboloid_env)
    test_distance_and_reward_logic(paraboloid_env)
    test_reset_options_parsing(paraboloid_env)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY.")
    print("="*60)

if __name__ == "__main__":
    main()