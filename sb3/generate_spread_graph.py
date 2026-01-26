import argparse
import glob
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import cbor2
import numpy as np
import matplotlib.pyplot as plt
import vmas

# ---------- Plot style ----------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

# ---------- Method display & colors ----------
METHOD_DISPLAY = {
    # Custom
    'infomarl': 'Ours',
    'infomarl_rnd': 'Ours (rnd)',
    'infomarl_nomask': 'Ours (no mask)',
    'gsa': 'GSA',
    'ph-marl': 'pH-MARL',
    
    # BenchMARL Baselines
    'benchmarl_mappo': 'MAPPO',
    'benchmarl_qmix': 'QMIX',
    'benchmarl_ippo': 'IPPO',
    'benchmarl_masac': 'MASAC',

    # CPM Baselines
    'cpm': 'CPM',
}

METHOD_COLORS = {
    'Ours':          '#2E86AB',  # Blue
    'Ours (rnd)':    '#17becf',  # Cyan
    'Ours (no mask)':'#2ca02c',  # Green
    'GSA':           '#F18F01',  # Orange
    'pH-MARL':       '#A23B72',  # Purple
    
    # Baselines
    'MAPPO':         '#7f7f7f',  # Gray
    'QMIX':          '#d62728',  # Red
    'IPPO':          '#9467bd',  # Violet
    'MASAC':         '#8c564b',  # Brown
    'CPM':           '#e377c2',  # Pink
}

PLOT_ORDER = [
    'Ours', 
    'MAPPO', 
    'QMIX', 
    'IPPO', 
    'MASAC',
    'GSA', 
    'pH-MARL',
    'CPM',
]

# ---------- Core per-file compute (vectorized single-loop) ----------
def fast_compute_rewards(path: str, normalize_per_robot: bool = True):
    """
    Reads a .dat recording (CBOR format), reconstructs the VMAS env, 
    replays positions, and computes the ground-truth reward.
    """
    with open(path, "rb") as f:
        # Header 1: Env Kwargs
        try:
            env_kwargs = cbor2.load(f)
        except Exception as e:
            print(f"Error loading header from {path}: {e}")
            return None, None

        scenario = env_kwargs.get("scenario", "simple_spread")
        if scenario != "simple_spread":
             # Skip mismatched files silently
            return None, None

        n_agents = int(env_kwargs.get("n_agents", 3))
        num_envs = int(env_kwargs.get("num_envs", 64))
        
        # Header 2: Meta (Model Name, Max Steps)
        meta = cbor2.load(f)
        max_steps = int(meta.get("max_steps", 400))
        raw_model_name = str(meta.get("model", "unknown"))
        
        # Normalize model name for display
        method_label = METHOD_DISPLAY.get(raw_model_name, raw_model_name)

        # Reconstruct Environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            num_envs=num_envs,
            continuous_actions=True, 
            max_steps=max_steps,
            seed=env_kwargs.get("seed", 42),
            device=device,
            terminated_truncated=False,
        )
        
        agents = env.agents
        landmarks = env.world.landmarks

        # Tensor to accumulate rewards: [num_envs]
        episodic_reward = torch.zeros(num_envs, device=device, dtype=torch.float32)

        steps_read = 0
        
        # Stream frames
        while steps_read < max_steps:
            try:
                frame = cbor2.load(f)
            except EOFError:
                break
            
            # 1. Update Agent Positions
            agent_data = frame.get("agent_data", {})
            for i, agent in enumerate(agents):
                name = agent.name
                pos_list = agent_data.get(name)
                if pos_list is not None:
                    agent.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            # 2. Update Landmark Positions
            landmark_data = frame.get("landmarks", {})
            for j, lm in enumerate(landmarks):
                # Handle inconsistent naming (landmark 0 vs landmark_0)
                name_variants = [lm.name, f"landmark {j}", f"landmark_{j}"]
                pos_list = None
                for nv in name_variants:
                    if nv in landmark_data:
                        pos_list = landmark_data[nv]
                        break
                
                if pos_list is not None:
                    lm.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            # 3. Compute Reward
            r = env.scenario.reward(agents[0]) 
            episodic_reward += r
            steps_read += 1

    # Normalize
    if normalize_per_robot:
        episodic_reward = episodic_reward / float(n_agents)

    return episodic_reward, {
        "scenario": scenario,
        "n_agents": n_agents,
        "num_envs": num_envs,
        "max_steps": steps_read,
        "method_label": method_label,
        "raw_model": raw_model_name
    }

# ---------- Aggregation ----------
def aggregate_directory(data_dir: str) -> Dict:
    # Look for *_spread_*.dat
    files = sorted(glob.glob(os.path.join(data_dir, "*_spread_*.dat")))
    if not files:
        print(f"No simple_spread files found in {data_dir}")
        return {}

    results = {}
    print(f"Processing {len(files)} files for Simple Spread...")

    for path in files:
        try:
            episodic_rewards, meta = fast_compute_rewards(path, normalize_per_robot=True)
        except Exception as e:
            print(f"  [Error] {Path(path).name}: {e}")
            continue

        if episodic_rewards is None: 
            continue

        method = meta["method_label"]
        n_agents = meta["n_agents"]
        
        rewards_cpu = episodic_rewards.detach().cpu().numpy()
        mean_r = float(np.mean(rewards_cpu))
        std_r = float(np.std(rewards_cpu))
        
        if method not in results:
            results[method] = {}
            
        results[method][n_agents] = {
            "mean": mean_r,
            "std": std_r,
            "n_episodes": meta["num_envs"],
            "file": Path(path).name
        }

        print(f"  [OK] {method.ljust(15)} | N={n_agents} | Mean R: {mean_r:.2f} +/- {std_r:.2f}")

    # Save JSON/CSV
    with open("spread_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    save_csv(results, methods=PLOT_ORDER)
    return results

def save_csv(results: Dict, methods: List[str]) -> None:
    # Filter active methods
    active_methods = [m for m in methods if m in results]
    # Add extra methods not in PLOT_ORDER
    for m in results.keys():
        if m not in active_methods:
            active_methods.append(m)

    agent_counts = sorted({n for m in active_methods if m in results for n in results[m].keys()})
    
    with open("spread_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["n_agents"]
        for m in active_methods:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        for n in agent_counts:
            row = [n]
            for m in active_methods:
                if n in results[m]:
                    row += [results[m][n]["mean"], results[m][n]["std"]]
                else:
                    row += [0.0, 0.0]
            w.writerow(row)
    print("💾 Saved spread_results.csv")

# ---------- Plotting ----------
def plot_results(results: Dict) -> None:
    active_methods = [m for m in PLOT_ORDER if m in results]
    for m in results.keys():
        if m not in active_methods:
            active_methods.append(m)

    all_agent_counts = set()
    for method in results:
        all_agent_counts.update(results[method].keys())
    
    if not all_agent_counts:
        print("No data to plot.")
        return
        
    agent_counts = sorted(list(all_agent_counts))
    
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(agent_counts))
    
    num_methods = len(active_methods)
    total_width = 0.8
    bar_width = total_width / num_methods
    
    for i, m in enumerate(active_methods):
        means, stds = [], []
        for n in agent_counts:
            if n in results[m]:
                means.append(results[m][n]["mean"])
                stds.append(results[m][n]["std"])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - num_methods / 2) * bar_width + (bar_width / 2)
        
        ax.bar(
            x + offset, 
            means, 
            bar_width, 
            yerr=stds, 
            capsize=4, 
            label=m,
            color=METHOD_COLORS.get(m, "#777777"),
            edgecolor="black", 
            linewidth=0.6, 
            alpha=0.9
        )

    ax.set_xlabel("Number of Agents", fontsize=12)
    ax.set_ylabel("Normalized Episodic Reward", fontsize=12)
    ax.set_title("Simple Spread (Cooperative Navigation)", fontsize=13, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.legend(loc="upper right", frameon=True, fontsize=10, ncol=1)

    plt.tight_layout()
    plt.savefig("fig_simple_spread.svg", format="svg", bbox_inches="tight")
    print("\n🖼️ Saved fig_simple_spread.svg")
    plt.savefig("fig_simple_spread.png", format="png", dpi=300, bbox_inches="tight")
    print("🖼️ Saved fig_simple_spread.png")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="eval_data", help="Directory with *_spread_*.dat files")
    args = parser.parse_args()

    results = aggregate_directory(args.eval_dir)
    if results:
        plot_results(results)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()