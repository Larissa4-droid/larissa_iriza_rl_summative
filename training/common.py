"""Shared helpers for all four algorithms: TorchRL environment, logging,
evaluation on unseen seasons, and model saving/loading."""
from __future__ import annotations

import os
os.environ.setdefault("RL_WARNINGS", "False")
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrl.envs import GymWrapper, TransformedEnv, RewardSum
from torchrl.modules import MLP

from environment.custom_env import MarketLinkEnv, OBS_NAMES, ACTION_NAMES

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
N_OBS = len(OBS_NAMES)
N_ACT = len(ACTION_NAMES)
EVAL_SEEDS = list(range(10_000, 10_020))   # 20 seasons never used in training


def make_env(seed: int = 0, scenario: str = "normal"):
    """The custom Gymnasium environment wrapped for TorchRL (one-hot actions)."""
    env = GymWrapper(MarketLinkEnv(scenario=scenario), device="cpu")
    env = TransformedEnv(env, RewardSum())          # adds 'episode_reward'
    env.set_seed(seed)
    return env


def mlp(out_features: int, hidden: list[int]) -> nn.Module:
    return MLP(in_features=N_OBS, out_features=out_features, num_cells=list(hidden), activation_class=nn.Tanh)


class RunLog:
    """Collects training curves for one run and saves them as JSON."""

    def __init__(self, algo: str, run_id: int, config: dict):
        self.algo, self.run_id, self.config = algo, run_id, config
        self.episode_rewards: list[float] = []
        self.curves: dict[str, list] = {}       # name -> list of (frames, value)
        self.t0 = time.time()

    def add_episodes(self, td):
        done = td["next", "done"].squeeze(-1)
        if done.any():
            self.episode_rewards += td["next", "episode_reward"].squeeze(-1)[done].tolist()

    def add(self, name: str, frames: int, value: float):
        self.curves.setdefault(name, []).append((int(frames), float(value)))

    def last_mean(self, n: int = 20) -> float:
        return float(np.mean(self.episode_rewards[-n:])) if self.episode_rewards else float("nan")

    def save(self, evaluation: dict, frames: int) -> dict:
        out = dict(algo=self.algo, run=self.run_id, config=self.config, frames=frames,
                   minutes=round((time.time() - self.t0) / 60, 2), episode_rewards=self.episode_rewards,
                   curves=self.curves, evaluation=evaluation)
        path = LOG_DIR / self.algo
        path.mkdir(parents=True, exist_ok=True)
        (path / f"run_{self.run_id:02d}.json").write_text(json.dumps(out))
        return out


# ---------------------------------------------------------------------------
# Evaluation: greedy policy on 20 unseen seasons
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net: nn.Module, seeds=EVAL_SEEDS, scenario: str = "normal", **options) -> dict:
    """Runs the greedy policy (argmax of Q-values or of policy logits)."""
    return evaluate_policy(lambda obs, env: int(net(torch.as_tensor(obs)).argmax()), seeds, scenario, **options)


def evaluate_policy(policy, seeds=EVAL_SEEDS, scenario: str = "normal", **options) -> dict:
    rewards, waste, net_rwf, failed, kigali, platform = [], [], [], 0, [], []
    for s in seeds:
        env = MarketLinkEnv(scenario=scenario, **options)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            done = term or trunc
        t = env.totals
        rewards.append(t["reward"])
        waste.append(t["wasted_kg"] / max(t["harvested_kg"], 1))
        net_rwf.append(t["income"] - t["costs"])
        kigali.append(t["kigali_kg"])
        platform.append(t["platform_kg"])
        failed += env.day < 45
    return dict(mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
                waste_pct=float(100 * np.mean(waste)), net_income_rwf=float(np.mean(net_rwf)),
                kigali_kg=float(np.mean(kigali)), platform_kg=float(np.mean(platform)),
                failed_pct=float(100 * failed / len(seeds)))


# ---------------------------------------------------------------------------
# Saving / loading models
# ---------------------------------------------------------------------------
def save_model(net: nn.Module, subdir: str, algo: str, config: dict, evaluation: dict, run_id: int) -> Path:
    path = MODEL_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"{algo}_run_{run_id:02d}.pt"
    torch.save(dict(state_dict=net.state_dict(), algo=algo, hidden=config["hidden"],
                    config=config, run=run_id, evaluation=evaluation), f)
    return f


def load_policy_net(path) -> tuple[nn.Module, dict]:
    """Loads a saved Q-network (DQN) or policy network (PG); argmax gives the action."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = mlp(N_ACT, ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def write_sweep_table(algo: str, results: list[dict], columns: dict) -> pd.DataFrame:
    """columns maps table header -> config key. Adds the evaluation results."""
    rows = []
    for r in results:
        row = {"Run": r["run"]}
        for header, key in columns.items():
            row[header] = r["config"][key]
        e = r["evaluation"]
        row["Mean reward (eval)"] = round(e["mean_reward"], 2)
        row["Std"] = round(e["std_reward"], 2)
        row["Waste %"] = round(e["waste_pct"], 1)
        row["Net income (M RWF)"] = round(e["net_income_rwf"] / 1e6, 2)
        row["Last-20 train reward"] = round(float(np.mean(r["episode_rewards"][-20:])), 2) if r["episode_rewards"] else None
        rows.append(row)
    df = pd.DataFrame(rows)
    LOG_DIR.mkdir(exist_ok=True)
    df.to_csv(LOG_DIR / f"{algo}_sweep.csv", index=False)
    print(f"\n{algo.upper()} sweep\n", df.to_string(index=False))
    return df
