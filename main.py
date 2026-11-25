"""Watch the best trained agent sell a tomato cooperative's harvest for one season.

    uv run main.py                          
    uv run main.py --algo ppo               
    uv run main.py --scenario price_glut    
    uv run main.py --random                 
    uv run main.py --no-render              
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import torch

from environment.custom_env import ACTION_NAMES, SCENARIOS, MarketLinkEnv

ROOT = Path(__file__).resolve().parent
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


def load_agent(algo: str | None):
    from training.common import load_policy_net
    best_file = ROOT / "logs" / "best_models.json"
    if not best_file.exists():
        raise SystemExit("No trained models found. Train first: see README (uv run python -m training.dqn_training ...).")
    best = json.loads(best_file.read_text())
    algo = algo or best["best_overall"]
    info = best["models"][algo]
    net, ckpt = load_policy_net(ROOT / info["model"])
    print(f"Loaded {NAMES[algo]} run {info['run']} ({info['model']}), "
          f"evaluation reward {info['evaluation']['mean_reward']:.2f} on 20 unseen seasons")
    return algo, net


def run(args):
    if args.random:
        algo, net = "random", None
        print("Running an untrained agent that picks random actions")
    else:
        algo, net = load_agent(args.algo)
    env = MarketLinkEnv(render_mode=None if args.no_render else "human", scenario=args.scenario)
    if not args.no_render:
        env.metadata["render_fps"] = args.fps

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\n=== Season {ep + 1} | scenario: {args.scenario} | seed {args.seed + ep} ===")
        print(f"{'day':>3}  {'action':<17} {'reward':>7} | {'stock kg':>8} {'cold kg':>7} {'Kigali':>6} "
              f"{'forecast':>8} {'orders':>6} | {'income RWF':>11} {'wasted kg':>9}")
        done = False
        while not done:
            with torch.no_grad():
                action = env.action_space.sample() if net is None else int(net(torch.as_tensor(obs)).argmax())
            day = env.day + 1
            forecast, kigali = env.forecast, env.price_kigali
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            print(f"{day:>3}  {ACTION_NAMES[action]:<17} {reward:>+7.2f} | {info['stock_kg']:>8,.0f} "
                  f"{info['cold_kg']:>7,.0f} {kigali:>6,.0f} {forecast:>8,.0f} {info['orders']:>6} | "
                  f"{info['income_rwf']:>11,.0f} {info['wasted_kg']:>9,.0f}   {env.last_event}")
        t = env.totals
        print(f"\nSeason {ep + 1} done ({algo}): reward {t['reward']:.2f} | income {t['income']:,.0f} RWF | "
              f"costs {t['costs']:,.0f} RWF | net {t['income'] - t['costs']:,.0f} RWF")
        print(f"Sold: intermediary {t['intermediary_kg']:,.0f} kg, local market {t['local_kg']:,.0f} kg, "
              f"Kigali {t['kigali_kg']:,.0f} kg, platform buyers {t['platform_kg']:,.0f} kg | "
              f"wasted {t['wasted_kg']:,.0f} kg of {t['harvested_kg']:,.0f} kg harvested "
              f"({100 * t['wasted_kg'] / max(t['harvested_kg'], 1):.1f}%)")
    if not args.no_render:
        import time
        time.sleep(4)       
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MarketLink RL: watch an agent sell a cooperative's tomato harvest")
    p.add_argument("--algo", choices=list(NAMES), default=None)
    p.add_argument("--scenario", choices=list(SCENARIOS), default="normal")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=30_000, help="season seed (30000+ never used in training)")
    p.add_argument("--fps", type=int, default=3, help="simulated days per second in the window")
    p.add_argument("--random", action="store_true")
    p.add_argument("--no-render", action="store_true")
    run(p.parse_args())
