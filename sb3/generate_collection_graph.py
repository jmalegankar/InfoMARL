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

}

PLOT_ORDER = [
    'MAPPO', 
    'QMIX', 
    'IPPO', 
    'MASAC',
    'GSA', 
    'pH-MARL',
    'Ours', 
    'Ours (rnd)',
    'Ours (no mask)',]

# ---------- Core per-file compute (vectorized single-loop) ----------
def fast_compute_rewards_food(path: str, normalize_per_robot: bool = True):
    """
    Returns:
      episodic_per_env: torch.Tensor [num_envs] per-robot episodic returns
      meta: dict with fields (scenario, n_agents, n_food, num_envs, max_steps, method, device)
    """
    with open(path, "rb") as f:
        # Chunk 1: env config
        try:
            cfg = cbor2.load(f)
        except Exception as e:
            print(f"Error loading header from {path}: {e}")
            return None, None

        scenario = cfg.get("scenario", "food_collection")
        if scenario != "food_collection":
            # Silently skip non-food files if they accidentally get matched
            return None, None

        n_agents = int(cfg.get("n_agents", 3))
        # Default n_food to n_agents if not present
        n_food = int(cfg.get("n_food", n_agents))
        
        num_envs = int(cfg.get("num_envs", 64))

        # Device resolution
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Chunk 2: meta / max steps / model name
        meta2 = cbor2.load(f)
        max_steps = int(meta2.get("max_steps", 400))
        raw_model = str(meta2.get("model", "unknown"))
        method = METHOD_DISPLAY.get(raw_model, raw_model)

        # Build env
        # Note: We relax the n_food check to allow generalization tests if needed,
        # but the standard plot usually assumes n_food=n_agents.
        env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            n_food=n_food,
            num_envs=num_envs,
            continuous_actions=True,
            max_steps=max_steps,
            seed=cfg.get("seed", 42),
            device=device,
            terminated_truncated=bool(cfg.get("terminated_truncated", False)),
        )
        
        agents = env.agents
        landmarks = env.world.landmarks 

        episodic = torch.zeros(num_envs, device=device, dtype=torch.float32)

        steps_read = 0
        while steps_read < max_steps:
            try:
                frame = cbor2.load(f)
            except EOFError:
                break

            agent_data = frame.get("agent_data", {})
            food_data = frame.get("landmarks", {})

            # Set agent positions
            for i, agent in enumerate(agents):
                name = agent.name # e.g. "agent_0"
                pos_list = agent_data.get(name, None)
                if pos_list is not None:
                    agent.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            # Set food positions
            for j, food in enumerate(landmarks):
                name = food.name # e.g. "landmark_0" or "food_0" depending on VMAS version
                pos_list = food_data.get(name, None)
                if pos_list is not None:
                    food.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            r = env.scenario.reward(agents[0])
            episodic += r
            steps_read += 1

    # Per-robot normalization
    if normalize_per_robot:
        episodic = episodic / float(n_agents)

    return episodic, {
        "scenario": scenario,
        "n_agents": n_agents,
        "n_food": n_food,
        "num_envs": num_envs,
        "max_steps": steps_read,
        "method": method,
        "device": str(device),
    }

# ---------- Aggregation & I/O ----------
def aggregate_directory(eval_dir: str) -> Dict:
    """
    Iterate over *_collection_*.dat, keep only scenario==food_collection.
    Returns nested dict: results[method][n_agents] = {mean, std, min, max, n_episodes}
    """
    # Pattern match for collection files
    files = sorted(glob.glob(os.path.join(eval_dir, "*_collection_*.dat")))

    #bascially only want files where n_agents == n_food
    filtered_files = []
    for file in files:
        try:
            with open(file, "rb") as f:
                cfg = cbor2.load(f)
                n_agents = int(cfg.get("n_agents", 3))
                n_food = int(cfg.get("n_food", n_agents))
                if n_agents == n_food:
                    filtered_files.append(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    files = filtered_files

    if not files:
        print(f"No food_collection files found in {eval_dir}")
        return {}

    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    print(f"Processing {len(files)} files for Food Collection...")

    for path in files:
        try:
            episodic, meta = fast_compute_rewards_food(path, normalize_per_robot=True)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue

        if episodic is None:
            continue

        method = meta["method"]
        n_agents = meta["n_agents"]
        
        ep = episodic.detach().float().cpu()
        mean = float(ep.mean().item())
        std = float(ep.std().item())
        
        if method not in results:
            results[method] = {}
            
        results[method][n_agents] = {
            "mean": mean,
            "std": std,
            "n_episodes": int(ep.numel()),
            "file": os.path.basename(path),
        }

        print(f"  [OK] {method.ljust(15)} | N={n_agents} | Mean R: {mean:.2f} +/- {std:.2f}")

    # Save JSON
    with open("food_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Saved food_results.json")

    # Also CSV
    save_csv(results, methods=PLOT_ORDER)

    return results

def save_csv(results: Dict, methods: List[str]) -> None:
    # Filter methods to only those present in results
    active_methods = [m for m in methods if m in results]
    # Add any extra methods found but not in PLOT_ORDER
    for m in results.keys():
        if m not in active_methods:
            active_methods.append(m)

    agent_counts = sorted({n for m in active_methods if m in results for n in results[m].keys()})
    
    with open("food_results.csv", "w", newline="") as f:
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
    print("💾 Saved food_results.csv")

# ---------- Plot ----------
def plot_results(results: Dict) -> None:
    # Filter to active methods
    active_methods = [m for m in PLOT_ORDER if m in results]
    # Add leftovers
    for m in results.keys():
        if m not in active_methods:
            active_methods.append(m)

    # collect Ns
    agent_counts = sorted({n for m in active_methods if m in results for n in results[m].keys()})
    if not agent_counts:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(agent_counts))
    
    # Dynamic width calculation
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
                means.append(0.0)
                stds.append(0.0)

        # Offset bars
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
    ax.set_title("Food Collection", fontsize=13, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig("fig_food_collection.svg", format="svg", bbox_inches="tight")
    print("\n🖼️ Saved fig_food_collection.svg")
    plt.savefig("fig_food_collection.png", format="png", dpi=300, bbox_inches="tight")
    print("🖼️ Saved fig_food_collection.png")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate & plot VMAS food_collection evaluations")
    ap.add_argument("--eval_dir", type=str, default="eval_data1", help="Directory with *_collection_*.dat files")
    args = ap.parse_args()

    results = aggregate_directory(args.eval_dir)
    if not results:
        print("No results to visualize.")
        return
    plot_results(results)

if __name__ == "__main__":
    main()