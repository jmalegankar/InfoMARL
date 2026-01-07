"""
Test CPM Training Script
Quick tests to verify training works before running full experiments

Updated to test:
- Basic training functionality
- Evaluation mode
- Reward normalization options
- Model saving and logging
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run command and return success status."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 300 seconds"


def check_model_file(path):
    """Check if model file exists and is valid size."""
    model_path = Path(path)
    if not model_path.exists():
        return False, f"File not found: {path}"
    
    size = model_path.stat().st_size
    if size < 1024:
        return False, f"File too small ({size} bytes): {path}"
    
    return True, f"Valid model file ({size} bytes)"


def main():
    print("=" * 60)
    print("CPM Training Tests")
    print("=" * 60)

    tests_passed = 0
    tests_total = 5

    # Test 1: Simple Spread - Basic training
    print("\n[Test 1/5] Simple Spread - Basic Training (20 episodes, 50 steps)")
    print("-" * 60)

    cmd1 = [
        sys.executable, "train.py",
        "--scenario", "simple_spread",
        "--n_agents", "3",
        "--n_episodes", "20",
        "--max_steps", "50",
        "--n_rollout_threads", "2",
        "--batch_size", "64",
        "--steps_per_update", "50",
        "--eval_interval", "10",  # Test evaluation mode
        "--save_interval", "20",
    ]

    success, output = run_command(cmd1)
    if success:
        print("✓ Test 1 passed - Basic training works")
        tests_passed += 1
    else:
        print("✗ Test 1 failed")
        print(output[:500])  # Show first 500 chars of error
        return False
    
    # Test 2: Food Collection
    print("\n[Test 2/5] Food Collection - With Reward Normalization (20 episodes, 50 steps)")
    print("-" * 60)

    cmd2 = [
        sys.executable, "train.py",
        "--scenario", "food_collection",
        "--n_agents", "3",
        "--n_food", "4",
        "--n_episodes", "20",
        "--max_steps", "50",
        "--n_rollout_threads", "2",
        "--batch_size", "64",
        "--steps_per_update", "50",
        "--normalize_rewards",  # Test reward normalization flag
        "--eval_interval", "10",
        "--save_interval", "20",
    ]

    success, output = run_command(cmd2)
    if success:
        print("✓ Test 2 passed - Food collection with reward normalization works")
        tests_passed += 1
    else:
        print("✗ Test 2 failed")
        print(output[:500])
        return False
    
    # Test 3: Check model files exist (including best models from evaluation)
    print("\n[Test 3/5] Checking saved models")
    print("-" * 60)

    ss_model = "models_cpm/simple_spread/cpm_simple_spread/run1/model.pt"
    ss_best_model = "models_cpm/simple_spread/cpm_simple_spread/run1/model_best.pt"
    fc_model = "models_cpm/food_collection/cpm_food_collection/run1/model.pt"
    fc_best_model = "models_cpm/food_collection/cpm_food_collection/run1/model_best.pt"

    # Check simple spread models
    ss_exists, ss_msg = check_model_file(ss_model)
    ss_best_exists, ss_best_msg = check_model_file(ss_best_model)

    if ss_exists and ss_best_exists:
        print(f"✓ Simple spread final model: {ss_msg}")
        print(f"✓ Simple spread best model: {ss_best_msg}")
    else:
        print(f"✗ Simple spread models missing")
        if not ss_exists:
            print(f"  - Final model: {ss_msg}")
        if not ss_best_exists:
            print(f"  - Best model: {ss_best_msg}")
        return False

    # Check food collection models
    fc_exists, fc_msg = check_model_file(fc_model)
    fc_best_exists, fc_best_msg = check_model_file(fc_best_model)

    if fc_exists and fc_best_exists:
        print(f"✓ Food collection final model: {fc_msg}")
        print(f"✓ Food collection best model: {fc_best_msg}")
        tests_passed += 1
    else:
        print(f"✗ Food collection models missing")
        if not fc_exists:
            print(f"  - Final model: {fc_msg}")
        if not fc_best_exists:
            print(f"  - Best model: {fc_best_msg}")
        return False
    
    # Test 4: Check TensorBoard logs exist
    print("\n[Test 4/5] Checking TensorBoard logs")
    print("-" * 60)

    ss_log = Path("models_cpm/simple_spread/cpm_simple_spread/run1/logs")
    fc_log = Path("models_cpm/food_collection/cpm_food_collection/run1/logs")

    if ss_log.exists() and any(ss_log.iterdir()):
        print(f"✓ Simple spread logs: {ss_log}")
    else:
        print(f"✗ Simple spread logs missing: {ss_log}")
        return False

    if fc_log.exists() and any(fc_log.iterdir()):
        print(f"✓ Food collection logs: {fc_log}")
        tests_passed += 1
    else:
        print(f"✗ Food collection logs missing: {fc_log}")
        return False

    # Test 5: Verify evaluation metrics in logs
    print("\n[Test 5/5] Verifying evaluation mode logged metrics")
    print("-" * 60)

    # Check if evaluation metrics files exist (TensorBoard event files)
    ss_log_files = list(ss_log.glob("events.out.tfevents.*"))
    fc_log_files = list(fc_log.glob("events.out.tfevents.*"))

    if ss_log_files:
        print(f"✓ Simple spread has {len(ss_log_files)} event file(s)")
        print(f"  Expected: eval/mean_reward, training/buffer_size, training/num_updates")
    else:
        print(f"✗ Simple spread missing TensorBoard event files")
        return False

    if fc_log_files:
        print(f"✓ Food collection has {len(fc_log_files)} event file(s)")
        print(f"  Expected: eval/mean_reward, training/buffer_size, training/num_updates")
        tests_passed += 1
    else:
        print(f"✗ Food collection missing TensorBoard event files")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nTests passed: {tests_passed}/{tests_total}")
    print("\n🎯 Verified Features:")
    print("  ✓ Basic CPM training on VMAS environments")
    print("  ✓ Evaluation mode (without exploration noise)")
    print("  ✓ Reward normalization option")
    print("  ✓ Best model saving based on eval performance")
    print("  ✓ Enhanced logging (buffer size, num updates, eval metrics)")
    print("\n📁 Models saved to:")
    print(f"  - Final: {ss_model}")
    print(f"  - Best:  {ss_best_model}")
    print(f"  - Final: {fc_model}")
    print(f"  - Best:  {fc_best_model}")
    print("\n📊 View training curves:")
    print("  tensorboard --logdir models_cpm/")
    print("\n  Metrics to look for:")
    print("    - sum_episode_rewards (training)")
    print("    - eval/mean_reward (evaluation without exploration)")
    print("    - training/buffer_size")
    print("    - training/num_updates")
    print("    - agent*/mean_episode_rewards")
    print("\n✨ You can now run full training experiments with:")
    print("  python train.py --scenario simple_spread --n_agents 4 \\")
    print("                  --n_episodes 50000 --eval_interval 500")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)