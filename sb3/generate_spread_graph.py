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
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

# ---------- Method display & colors ----------
METHOD_DISPLAY = {
    'infomarl': 'Ours',
    'infomarl_rnd': 'Ours rnd mask',
    'infomarl_nomask': 'Ours w/o mask',
    'gsa': 'GSA',
    'ph-marl': 'pH-MARL',
}
METHOD_COLORS = {
    'Ours':   '#2E86AB',  # Blue
    'Ours rnd mask' : '#17becf',    # Cyan
    'Ours w/o mask': '#2ca02c',     # Green
    'GSA':    '#F18F01',  # Orange
    'pH-MARL':'#A23B72',  # Purple
}
PLOT_ORDER = ['Ours', 'Ours rnd mask', 'Ours w/o mask', 'GSA', 'pH-MARL']  # requested order

# ---------- Core per-file compute (vectorized single-loop) ----------
def fast_compute_rewards(path: str, normalize_per_robot: bool = True, normalize_per_robot_square: bool = False,):
    """
    Returns:
      episodic_per_env: torch.Tensor [num_envs] per-robot episodic returns
      meta: dict with fields (scenario, n_agents, num_envs, max_steps, model)
    Raises:
      ValueError if file is not simple_spread.
    """
    with open(path, "rb") as f:
        # Chunk 1: env config
        cfg = cbor2.load(f)
        scenario = cfg.get("scenario", "simple_spread")
        if scenario != "simple_spread":
            raise ValueError(f"{path}: scenario '{scenario}' != 'simple_spread' (skipping).")

        n_agents = int(cfg.get("n_agents", 3))
        num_envs = int(cfg.get("num_envs", 64))
        dev_str = str(cfg.get("device", "cpu"))

        # Device resolution
        device = torch.device("cuda" if (dev_str == "cuda" and torch.cuda.is_available()) else "cpu")

        # Chunk 2: meta / max steps / model name
        meta2 = cbor2.load(f)
        max_steps = int(meta2.get("max_steps", 400))
        raw_model = str(meta2.get("model", "unknown")).lower()
        method = METHOD_DISPLAY.get(raw_model, raw_model.upper())

        env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            num_envs=num_envs,
            continuous_actions=True,
            max_steps=max_steps,
            seed=cfg.get("seed", 42),
            device=device,
            terminated_truncated=bool(cfg.get("terminated_truncated", False)),
        )
        world = env.world
        agents = env.agents
        landmarks = world.landmarks

        episodic = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # Single loop over steps; each chunk sets ALL positions at once
        steps_read = 0
        while steps_read < max_steps:
            try:
                frame = cbor2.load(f)
            except EOFError:
                break

            agent_data = frame.get("agent_data", {})
            landmark_data = frame.get("landmarks", {})

            for i, agent in enumerate(agents):
                name = f"agent_{i}"
                pos_list = agent_data.get(name, None)
                agent.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            for j, lm in enumerate(landmarks):
                name = f"landmark {j}"
                pos_list = landmark_data.get(name, None)
                lm.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

            r = env.scenario.reward(agents[0])
            # print(r.shape, episodic.shape)
            episodic += r
            steps_read += 1

    # Per-robot normalization
    if normalize_per_robot:
        episodic = episodic / float(n_agents)
    elif normalize_per_robot_square:
        episodic = episodic / float(n_agents ** 2)

    return episodic, {
        "scenario": scenario,
        "n_agents": n_agents,
        "num_envs": num_envs,
        "max_steps": steps_read,
        "method": method,
        "device": str(device),
    }

# ---------- Aggregation & I/O ----------
def aggregate_directory(eval_dir: str, device: str = "auto") -> Dict:
    """
    Iterate over *_spread_*.dat, keep only scenario==simple_spread, compute stats.
    Returns nested dict: results[method][n_agents] = {mean, std, min, max, n_episodes}
    """
    files = sorted([p for p in glob.glob(os.path.join(eval_dir, "*_spread_*.dat")) if not p.endswith("_results.dat")])
    if not files:
        print(f"No spread files found in {eval_dir}")
        return {}

    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for path in files:
        try:
            episodic, meta = fast_compute_rewards(path,  normalize_per_robot=True, normalize_per_robot_square=False)
        except ValueError as ve:
            # Not simple_spread; skip
            print(f"SKIP: {ve}")
            continue
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue

        method = meta["method"]
        n_agents = meta["n_agents"]
        ep = episodic.detach().float().cpu()
        mean = float(ep.mean().item()) 
        std = float(ep.std().item()) 
        min_v = float(ep.min().item()) 
        max_v = float(ep.max().item())

        if method not in results:
            results[method] = {}
        results[method][n_agents] = {
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "n_episodes": int(ep.numel()),
            "file": os.path.basename(path),
        }

        print(f"✓ {Path(path).name}: {method}, N={n_agents}, "
              f"mean={mean:.2f}±{std:.2f}, min={min_v:.2f}, max={max_v:.2f}, n={int(ep.numel())}")

    # Save JSON
    with open("spread_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Saved spread_results.json")

    # Also CSV
    save_csv(results, methods=PLOT_ORDER)

    return results

def save_csv(results: Dict, methods: List[str]) -> None:
    # Collect union of agent counts across selected methods
    agent_counts = sorted({n for m in methods if m in results for n in results[m].keys()})
    with open("spread_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["n_agents"]
        for m in methods:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        for n in agent_counts:
            row = [n]
            for m in methods:
                if m in results and n in results[m]:
                    row += [results[m][n]["mean"], results[m][n]["std"]]
                else:
                    row += [0.0, 0.0]
            w.writerow(row)
    print("💾 Saved spread_results.csv")

# ---------- Plot ----------
def plot_results(results: Dict) -> None:
    methods = PLOT_ORDER
    # collect Ns
    agent_counts = sorted({n for m in methods if m in results for n in results[m].keys()})
    if not agent_counts:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(agent_counts))
    width = 0.15

    for i, m in enumerate(methods):
        means, stds = [], []
        for n in agent_counts:
            if m in results and n in results[m]:
                means.append(results[m][n]["mean"])
                stds.append(results[m][n]["std"])
            else:
                means.append(0.0)
                stds.append(0.0)

        bars = ax.bar(
            x + (i - 1) * width, means, width,
            yerr=stds, capsize=4,
            label=m,
            color=METHOD_COLORS.get(m, "#777777"),
            edgecolor="black", linewidth=0.6, alpha=0.92
        )

    ax.set_xlabel("Number of robots", fontsize=11)
    ax.set_ylabel("Normalized cumulative reward (R)", fontsize=11)
    ax.set_title("Simple-Spread", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend above plot
    ax.legend(loc="lower right", frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig("fig_simple_spread.svg", format="svg", bbox_inches="tight")
    print("🖼️ Saved fig_simple_spread.svg")
    plt.savefig("fig_simple_spread.png", format="png", dpi=300, bbox_inches="tight")
    print("🖼️ Saved fig_simple_spread.png")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate & plot VMAS simple_spread evaluations")
    ap.add_argument("--eval_dir", type=str, default="./eval_data", help="Directory with *_spread_*.dat files")
    ap.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    args = ap.parse_args()

    results = aggregate_directory(args.eval_dir, device=args.device)
    if not results:
        print("No results to visualize.")
        return
    plot_results(results)

if __name__ == "__main__":
    main()
