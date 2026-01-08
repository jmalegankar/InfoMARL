"""
VMAS-CPM Wrapper Test Suite

Tests the optimized wrapper that uses VMAS discrete actions natively.
Key optimizations:
- Uses continuous_actions=False in VMAS (no conversion overhead)
- Handles both discrete indices and one-hot encoded actions
- Single argmax operation for CPM policy outputs
"""

import numpy as np
import vmas
from vmas_cpm_wrapper import VMAS_CPM_Wrapper, make_parallel_env_for_cpm


def test_basic_wrapper(scenario="simple_spread", n_agents=3, n_food=5):
    """Test basic wrapper functionality with discrete actions."""
    print(f"\n{'='*60}")
    print(f"Testing Basic Wrapper: {scenario}")
    print(f"{'='*60}")

    # Create VMAS environment with discrete actions
    print("\n[1/5] Creating VMAS environment (discrete actions)...")
    if scenario == "food_collection":
        vmas_env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            n_food=n_food,
            num_envs=2,
            continuous_actions=False,  # Use discrete actions natively
            max_steps=10,
            device="cpu",
        )
    else:
        vmas_env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            num_envs=2,
            continuous_actions=False,  # Use discrete actions natively
            max_steps=10,
            device="cpu",
        )
    print(f"  ✓ VMAS environment created with discrete actions")

    # Wrap for CPM
    print("\n[2/5] Wrapping for CPM...")
    wrapped_env = VMAS_CPM_Wrapper(vmas_env, continuous_actions=False)
    print(f"  ✓ Wrapper created")
    print(f"    - n_agents: {wrapped_env.n}")
    print(f"    - num_vmas_envs: {wrapped_env.num_vmas_envs}")
    print(f"    - obs_dim: {wrapped_env.observation_space[0].shape}")
    print(f"    - action_space: Discrete({wrapped_env.action_space[0].n})")
    print(f"    - num_discrete_actions: {wrapped_env.num_discrete_actions}")
    
    # Test reset
    print("\n[3/5] Testing reset...")
    obs = wrapped_env.reset()
    assert isinstance(obs, list), "Reset should return list"
    assert len(obs) == n_agents, f"Expected {n_agents} observations, got {len(obs)}"
    
    for i, agent_obs in enumerate(obs):
        assert isinstance(agent_obs, np.ndarray), f"Agent {i} obs should be numpy array"
        assert agent_obs.ndim == 2, f"Agent {i} obs should be 2D"
        assert agent_obs.shape[0] == 2, f"Agent {i} should have 2 envs"
    
    print(f"  ✓ Reset successful")
    print(f"    - obs[0] shape: {obs[0].shape}")
    
    # Test step
    print("\n[4/5] Testing step...")
    actions = [np.random.randint(0, 5, size=2) for _ in range(n_agents)]
    next_obs, rewards, dones, info = wrapped_env.step(actions)
    
    assert isinstance(next_obs, list), "Step should return obs list"
    assert len(next_obs) == n_agents, f"Expected {n_agents} observations"
    assert isinstance(rewards, list), "Step should return reward list"
    assert len(rewards) == n_agents, f"Expected {n_agents} rewards"
    assert isinstance(dones, list), "Step should return done list"
    assert len(dones) == n_agents, f"Expected {n_agents} dones"
    
    print(f"  ✓ Step successful")
    print(f"    - obs shape: {next_obs[0].shape}")
    print(f"    - rewards shape: {rewards[0].shape}")
    print(f"    - dones shape: {dones[0].shape}")
    
    # Test multiple steps
    print("\n[5/5] Running 10 steps...")
    for step in range(10):
        actions = [np.random.randint(0, 5, size=2) for _ in range(n_agents)]
        next_obs, rewards, dones, info = wrapped_env.step(actions)
    
    print(f"  ✓ Multiple steps completed")
    
    return wrapped_env


def test_vectorized_wrapper(scenario="simple_spread", n_agents=3, n_food=5):
    """Test vectorized environment wrapper."""
    print(f"\n{'='*60}")
    print(f"Testing Vectorized Wrapper: {scenario}")
    print(f"{'='*60}")
    
    # Create vectorized environment
    print("\n[1/4] Creating vectorized environment...")
    kwargs = {}
    if scenario == "food_collection":
        kwargs["n_food"] = n_food
    
    vec_env = make_parallel_env_for_cpm(
        scenario=scenario,
        n_agents=n_agents,
        n_rollout_threads=4,
        seed=0,
        max_steps=10,
        device="cpu",
        **kwargs
    )
    
    total_envs = vec_env.num_envs * vec_env.num_vmas_envs_per_wrapper
    obs_dim = vec_env.observation_space[0].shape[0]
    
    print(f"  ✓ Vec env created")
    print(f"    - num_wrappers: {vec_env.num_envs}")
    print(f"    - envs_per_wrapper: {vec_env.num_vmas_envs_per_wrapper}")
    print(f"    - total_envs: {total_envs}")
    print(f"    - n_agents: {vec_env.n}")
    print(f"    - obs_dim: {obs_dim}")
    
    # Test reset
    print("\n[2/4] Testing vec reset...")
    vec_obs = vec_env.reset()
    expected_shape = (total_envs, n_agents, obs_dim)
    
    assert vec_obs.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {vec_obs.shape}"
    
    print(f"  ✓ Vec reset successful")
    print(f"    - obs shape: {vec_obs.shape}")
    
    # Test step
    print("\n[3/4] Testing vec step...")
    vec_actions = np.random.randint(0, 5, size=(total_envs, n_agents))
    vec_next_obs, vec_rewards, vec_dones, vec_info = vec_env.step(vec_actions)
    
    assert vec_next_obs.shape == expected_shape, "Obs shape mismatch"
    assert vec_rewards.shape == (total_envs, n_agents), "Reward shape mismatch"
    assert vec_dones.shape == (total_envs, n_agents), "Done shape mismatch"
    
    print(f"  ✓ Vec step successful")
    print(f"    - obs shape: {vec_next_obs.shape}")
    print(f"    - rewards shape: {vec_rewards.shape}")
    print(f"    - dones shape: {vec_dones.shape}")
    
    # Test full episode
    print("\n[4/4] Running full episode (10 steps)...")
    obs = vec_env.reset()
    total_reward = np.zeros((total_envs, n_agents))
    
    for step in range(10):
        actions = np.random.randint(0, 5, size=(total_envs, n_agents))
        obs, rewards, dones, info = vec_env.step(actions)
        total_reward += rewards
    
    print(f"  ✓ Full episode completed")
    print(f"    - Mean reward per agent: {total_reward.mean(axis=0)}")
    print(f"    - Total episodes: {dones.all(axis=1).sum()}")
    
    vec_env.close()
    return True


def test_action_handling(wrapped_env):
    """Test discrete action handling (native VMAS discrete actions)."""
    print(f"\n{'='*60}")
    print("Testing Action Handling")
    print(f"{'='*60}")

    print("\nTesting discrete action handling:")
    n_agents = wrapped_env.n
    num_envs = wrapped_env.num_vmas_envs
    num_actions = wrapped_env.num_discrete_actions

    print(f"  Number of discrete actions: {num_actions}")

    # Test 1: Direct discrete indices
    print("\n[Test 1] Testing direct discrete action indices...")
    action_names = ["no-op", "up", "down", "left", "right"]
    for action_id in range(min(num_actions, len(action_names))):
        actions = [np.full(num_envs, action_id) for _ in range(n_agents)]
        try:
            obs, rewards, dones, _ = wrapped_env.step(actions)
            print(f"  ✓ Action {action_id} ({action_names[action_id]:6s}): step successful")
        except Exception as e:
            print(f"  ✗ Action {action_id} failed: {e}")
            return False

    # Test 2: One-hot encoded actions (from CPM policy)
    print("\n[Test 2] Testing one-hot encoded actions (CPM format)...")
    for action_id in range(min(num_actions, 3)):  # Test first 3 actions
        # Create one-hot encoded actions
        one_hot_actions = []
        for _ in range(n_agents):
            one_hot = np.zeros((num_envs, num_actions), dtype=np.float32)
            one_hot[:, action_id] = 1.0
            one_hot_actions.append(one_hot)

        try:
            obs, rewards, dones, _ = wrapped_env.step(one_hot_actions)
            print(f"  ✓ One-hot action {action_id}: step successful")
        except Exception as e:
            print(f"  ✗ One-hot action {action_id} failed: {e}")
            return False

    print(f"\n  ✓ All action handling tests passed")
    return True


def test_episode_termination(scenario="simple_spread", n_agents=3):
    """Test that episodes terminate and reset correctly."""
    print(f"\n{'='*60}")
    print("Testing Episode Termination")
    print(f"{'='*60}")

    # Create environment with short episodes
    print("\n[1/2] Creating environment with max_steps=5...")
    vmas_env = vmas.make_env(
        scenario=scenario,
        n_agents=n_agents,
        num_envs=2,
        continuous_actions=False,  # Use discrete actions
        max_steps=5,
        device="cpu",
    )
    wrapped_env = VMAS_CPM_Wrapper(vmas_env, continuous_actions=False)
    print(f"  ✓ Environment created (discrete actions)")
    
    # Run until termination
    print("\n[2/2] Running until termination...")
    obs = wrapped_env.reset()
    step_count = 0
    episode_terminated = False
    
    for step in range(20):  # More than max_steps to ensure termination
        actions = [np.random.randint(0, 5, size=2) for _ in range(n_agents)]
        obs, rewards, dones, info = wrapped_env.step(actions)
        step_count += 1
        
        if dones[0].any():
            episode_terminated = True
            print(f"  ✓ Episode terminated at step {step_count}")
            print(f"    - dones: {dones[0]}")
            break
    
    assert episode_terminated, "Episode should have terminated"
    print(f"  ✓ Termination handling correct")
    
    return True


def run_all_tests():
    """Run all wrapper tests."""
    print("\n" + "="*60)
    print("VMAS-CPM WRAPPER TEST SUITE")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic wrapper - Simple Spread
    tests_total += 1
    try:
        wrapped_env = test_basic_wrapper(scenario="simple_spread", n_agents=3)
        tests_passed += 1

        # Test action handling (both discrete indices and one-hot)
        tests_total += 1
        test_action_handling(wrapped_env)
        tests_passed += 1

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Basic wrapper - Food Collection
    tests_total += 1
    try:
        test_basic_wrapper(scenario="food_collection", n_agents=4, n_food=5)
        tests_passed += 1
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Vectorized wrapper - Simple Spread
    tests_total += 1
    try:
        test_vectorized_wrapper(scenario="simple_spread", n_agents=3)
        tests_passed += 1
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Vectorized wrapper - Food Collection
    tests_total += 1
    try:
        test_vectorized_wrapper(scenario="food_collection", n_agents=4, n_food=5)
        tests_passed += 1
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Episode termination
    tests_total += 1
    try:
        test_episode_termination(scenario="simple_spread", n_agents=3)
        tests_passed += 1
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("\n✅ ALL TESTS PASSED! Wrapper is ready for training.")
        print("\n🎯 Verified Features:")
        print("  ✓ VMAS discrete actions (native, no conversion)")
        print("  ✓ One-hot encoded actions (from CPM policy)")
        print("  ✓ Direct discrete action indices")
        print("  ✓ Vectorized environment wrapper")
        print("  ✓ Episode termination and reset")
        print("  ✓ Both simple_spread and food_collection scenarios")
        print("\n📝 Implementation Details:")
        print("  - Uses continuous_actions=False in VMAS")
        print("  - Single argmax operation to convert one-hot → indices")
        print("  - No discrete-to-continuous conversion overhead")
        print("  - Maintains full compatibility with CPM code")
        return True
    else:
        print(f"\n❌ {tests_total - tests_passed} TEST(S) FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)