import argparse
import glob
import json
import csv
import os
import pickle
from pathlib import Path
from typing import Dict, List

import torch
import cbor2
import numpy as np
import matplotlib.pyplot as plt
import vmas

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
})

METHOD_DISPLAY = {
    'infomarl': 'Ours',
    'gsa': 'GSA',
    'ph-marl': 'pH-MARL',
    'benchmarl_mappo': 'MAPPO',
    'benchmarl_masac': 'MASAC',
    'benchmarl_qmix': 'QMIX',
}

METHOD_COLORS = {
    'Ours':    '#2E86AB',
    'GSA':     '#F18F01',
    'pH-MARL': '#A23B72',
    'MAPPO':   '#2CA02C',
    'MASAC':   '#D62728',
    'QMIX':    '#9467BD',
}

PLOT_ORDER = ['Ours', 'GSA', 'pH-MARL', 'MAPPO', 'MASAC', 'QMIX']


def fast_compute_rewards(path: str, normalize_per_robot: bool = True):
    """Compute rewards from pickle or CBOR file."""
    
    with open(path, "rb") as f:
        if path.endswith('.pkl'):
            # Pickle format
            data = pickle.load(f)
            cfg = data['env_kwargs']
            meta2 = data['meta']
            frames = data['frames']
            
            scenario = cfg.get("scenario", "simple_spread")
            if scenario != "simple_spread":
                raise ValueError(f"{path}: scenario '{scenario}' != 'simple_spread'")
            
            n_agents = int(cfg.get("n_agents", 3))
            num_envs = int(cfg.get("num_envs", 64))
            max_steps = int(meta2.get("max_steps", 400))
            raw_model = str(meta2.get("model", "unknown")).lower()
            method = METHOD_DISPLAY.get(raw_model, raw_model.upper())
            
            device = torch.device("cpu")
            
            env = vmas.make_env(
                scenario=scenario,
                n_agents=n_agents,
                num_envs=num_envs,
                continuous_actions=True,
                max_steps=max_steps,
                seed=cfg.get("seed", 42),
                device=device,
            )
            agents = env.agents
            landmarks = env.world.landmarks
            
            episodic = torch.zeros(num_envs, device=device, dtype=torch.float32)
            
            for frame in frames[:-1]:  # Skip final state frame
                agent_data = frame.get("agent_data", {})
                landmark_data = frame.get("landmarks", {})
                
                for i, agent in enumerate(agents):
                    name = f"agent_{i}"
                    pos_list = agent_data.get(name)
                    if pos_list:
                        agent.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)
                
                for j, lm in enumerate(landmarks):
                    name = f"landmark {j}"
                    pos_list = landmark_data.get(name)
                    if pos_list:
                        lm.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)
                
                r = env.scenario.reward(agents[0])
                episodic += r
            
            steps_read = len(frames) - 1
            
        else:
            cfg = cbor2.load(f)
            scenario = cfg.get("scenario", "simple_spread")
            if scenario != "simple_spread":
                raise ValueError(f"{path}: scenario '{scenario}' != 'simple_spread'")

            n_agents = int(cfg.get("n_agents", 3))
            num_envs = int(cfg.get("num_envs", 64))
            
            meta2 = cbor2.load(f)
            max_steps = int(meta2.get("max_steps", 400))
            raw_model = str(meta2.get("model", "unknown")).lower()
            method = METHOD_DISPLAY.get(raw_model, raw_model.upper())

            device = torch.device("cpu")

            env = vmas.make_env(
                scenario=scenario,
                n_agents=n_agents,
                num_envs=num_envs,
                continuous_actions=True,
                max_steps=max_steps,
                seed=cfg.get("seed", 42),
                device=device,
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
                landmark_data = frame.get("landmarks", {})

                for i, agent in enumerate(agents):
                    name = f"agent_{i}"
                    pos_list = agent_data.get(name)
                    if pos_list:
                        agent.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

                for j, lm in enumerate(landmarks):
                    name = f"landmark {j}"
                    pos_list = landmark_data.get(name)
                    if pos_list:
                        lm.state.pos = torch.tensor(pos_list, device=device, dtype=torch.float32)

                r = env.scenario.reward(agents[0])
                episodic += r
                steps_read += 1

    if normalize_per_robot:
        episodic = episodic / float(n_agents)

    return episodic, {
        "scenario": scenario,
        "n_agents": n_agents,
        "num_envs": num_envs,
        "max_steps": steps_read,
        "method": method,
    }


def aggregate_directory(eval_dir: str) -> Dict:
    # Find both .dat and .pkl files
    dat_files = glob.glob(os.path.join(eval_dir, "*_spread_*.dat"))
    pkl_files = glob.glob(os.path.join(eval_dir, "*_spread_*.pkl"))
    files = sorted([p for p in dat_files + pkl_files if not p.endswith("_results.dat")])
    
    if not files:
        print(f"No spread files found in {eval_dir}")
        return {}

    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for path in files:
        try:
            episodic, meta = fast_compute_rewards(path)
        except ValueError as ve:
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
              f"mean={mean:.2f}±{std:.2f}")

    with open("spread_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Saved spread_results.json")

    save_csv(results, methods=PLOT_ORDER)
    return results


def save_csv(results: Dict, methods: List[str]) -> None:
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


def plot_results(results: Dict) -> None:
    methods = [m for m in PLOT_ORDER if m in results]
    agent_counts = sorted({n for m in methods for n in results[m].keys()})
    
    if not agent_counts:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(agent_counts))
    width = 0.8 / len(methods)

    for i, m in enumerate(methods):
        means, stds = [], []
        for n in agent_counts:
            if n in results[m]:
                means.append(results[m][n]["mean"])
                stds.append(results[m][n]["std"])
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=m, color=METHOD_COLORS.get(m, "#777777"),
               edgecolor="black", linewidth=0.6, alpha=0.92)

    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Normalized cumulative reward")
    ax.set_title("Simple-Spread", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("fig_simple_spread.svg", bbox_inches="tight")
    plt.savefig("fig_simple_spread.png", dpi=300, bbox_inches="tight")
    print("🖼️ Saved fig_simple_spread.svg/png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="./recordings")
    args = ap.parse_args()

    results = aggregate_directory(args.eval_dir)
    if results:
        plot_results(results)


if __name__ == "__main__":
    main()