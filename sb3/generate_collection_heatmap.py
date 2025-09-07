#!/usr/bin/env python3
"""
food_infomarl_heatmap.py

Reads a directory of VMAS food_collection .dat (CBOR) files and:
  • Keeps ONLY InfoMARL runs (filenames containing "infomarl")
  • Validates each file is for scenario == "food_collection"
  • Builds a batched env, replays positions, recomputes rewards
  • Normalizes episodic return per robot (/ N_agents)
  • Aggregates stats by (n_agents, n_food)
  • Saves food_infomarl.json / .csv and a heatmap (SVG + PNG)

Usage:
  python food_infomarl_heatmap.py --eval_dir ./eval_data
"""

import argparse
import glob
import json
import csv
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import cbor2
import numpy as np
import matplotlib.pyplot as plt
import vmas


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})


def fast_compute_rewards_food(path: str):
    with open(path, "rb") as f:
        cfg = cbor2.load(f)
        scenario = cfg.get("scenario", "food_collection")
        if scenario != "food_collection":
            raise ValueError(f"{path}: scenario != food_collection")

        n_agents = int(cfg.get("n_agents", 3))
        n_food = int(cfg.get("n_food", n_agents))
        num_envs = int(cfg.get("num_envs", 64))
        dev_str = str(cfg.get("device", "cpu"))
        device = torch.device(
            "cuda" if (dev_str == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        meta2 = cbor2.load(f)
        max_steps = int(meta2.get("max_steps", 400))
        raw_model = str(meta2.get("model", "unknown")).lower()

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
        foods = env.world.landmarks

        episodic = torch.zeros(num_envs, device=device, dtype=torch.float32)
        steps_read = 0
        while steps_read < max_steps:
            try:
                frame = cbor2.load(f)
            except EOFError:
                break

            agent_data = frame.get("agent_data", {})
            food_data = frame.get("landmarks", {})

            for i, agent in enumerate(agents):
                name = f"agent_{i}"
                pos_list = agent_data.get(name, None)
                if pos_list is not None:
                    agent.state.pos = torch.tensor(
                        pos_list, device=device, dtype=torch.float32
                    )

            for j, food in enumerate(foods):
                name = f"food_{j}"
                pos_list = food_data.get(name, None)
                if pos_list is not None:
                    food.state.pos = torch.tensor(
                        pos_list, device=device, dtype=torch.float32
                    )

            r = env.scenario.reward(agents[0])
            episodic += r
            steps_read += 1

    episodic = episodic / float(n_agents)  # per-robot normalization

    return episodic, {
        "n_agents": n_agents,
        "n_food": n_food,
        "num_envs": num_envs,
        "max_steps": steps_read,
        "model": raw_model,
        "device": str(device),
    }


def aggregate_directory(eval_dir: str) -> Dict[Tuple[int, int], Dict]:
    files = sorted(
        [
            p
            for p in glob.glob(os.path.join(eval_dir, "*_collection_*.dat"))
            if "infomarl" in p.lower()
        ]
    )
    if not files:
        print(f"No InfoMARL food_collection files found in {eval_dir}")
        return {}

    results = {}
    for path in files:
        try:
            episodic, meta = fast_compute_rewards_food(path)
        except Exception as e:
            print(f"SKIP {path}: {e}")
            continue

        nA, nF = meta["n_agents"], meta["n_food"]
        ep = episodic.detach().cpu().numpy()
        mean, std = float(ep.mean()), float(ep.std())
        results[(nA, nF)] = {"mean": mean, "std": std, "n": int(ep.size)}

        print(
            f"✓ {Path(path).name}: A={nA}, F={nF}, mean={mean:.2f}±{std:.2f}, n={ep.size}"
        )

    results_json = {f"A={k[0]},F={k[1]}": v for k, v in results.items()}
    with open("food_infomarl.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("💾 Saved food_infomarl.json")

    all_agents = sorted({k[0] for k in results})
    all_food = sorted({k[1] for k in results})
    with open("food_infomarl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_food \\ n_agents"] + all_agents)
        for nF in all_food:
            row = [nF]
            for nA in all_agents:
                cell = results.get((nA, nF))
                row.append(cell["mean"] if cell else "")
            w.writerow(row)
    print("💾 Saved food_infomarl.csv")

    return results


def plot_heatmap(results: Dict[Tuple[int, int], Dict]):
    if not results:
        print("No results to plot.")
        return

    all_agents = sorted({k[0] for k in results})
    all_food = sorted({k[1] for k in results})
    mat = np.full((len(all_food), len(all_agents)), np.nan)
    std_mat = np.full_like(mat, np.nan)

    for i, nF in enumerate(all_food):
        for j, nA in enumerate(all_agents):
            cell = results.get((nA, nF))
            if cell:
                mat[i, j] = cell["mean"]
                std_mat[i, j] = cell["std"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean per-robot return")

    ax.set_xticks(range(len(all_agents)))
    ax.set_xticklabels(all_agents)
    ax.set_xlabel("Number of Robots")

    ax.set_yticks(range(len(all_food)))
    ax.set_yticklabels(all_food)
    ax.set_ylabel("Number of Food Items")

    ax.set_title("Food Collection Heatmap")

    # annotate with mean ± std
    for i in range(len(all_food)):
        for j in range(len(all_agents)):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i,j]:.0f}±{std_mat[i,j]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig("fig_food_infomarl_heatmap.svg", bbox_inches="tight")
    plt.savefig("fig_food_infomarl_heatmap.png", dpi=300, bbox_inches="tight")
    print("🖼️ Saved heatmap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="./eval_data")
    args = ap.parse_args()

    results = aggregate_directory(args.eval_dir)
    plot_heatmap(results)


if __name__ == "__main__":
    main()
