"""Builds every figure and table for the report from the saved training logs.

    uv run python -m training.analysis

Outputs (assets/ and logs/):
  cumulative_rewards.png   training reward per season, best run of each method
  training_stability.png   DQN loss curve and policy entropy of the PG methods
  convergence.png          moving-average reward vs seasons, with convergence points
  sweep_overview.png       evaluation reward of all 10 runs per method
  generalization.png       best models on unseen start conditions / market scenarios
  logs/best_models.json    which model main.py loads
  logs/*.csv               sweep tables, convergence, generalisation, behaviour
"""
from __future__ import annotations

import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from environment.custom_env import ACTION_NAMES, SCENARIOS, MarketLinkEnv, intermediary_policy, rule_based_policy
from training.common import EVAL_SEEDS, LOG_DIR, MODEL_DIR, ROOT, evaluate, evaluate_policy, load_policy_net

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#1f6fb2", "reinforce": "#d1495b", "ppo": "#2e8b57", "a2c": "#e08e0b"}
SUBDIR = {"dqn": "dqn", "reinforce": "pg", "ppo": "pg", "a2c": "pg"}
GEN_SEEDS = list(range(20_000, 20_020))      # different from training and from EVAL_SEEDS
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def load_runs(algo):
    return [json.loads(p.read_text()) for p in sorted((LOG_DIR / algo).glob("run_*.json"))]


def moving_avg(x, n=50):
    x = np.asarray(x, dtype=float)
    if len(x) < n:
        n = max(1, len(x))
    return np.convolve(x, np.ones(n) / n, mode="valid")


def convergence_episode(rewards, n=50, tol=0.95):
    """First season after which the 50-season moving average stays at >= 95% of its final level."""
    ma = moving_avg(rewards, n)
    final = np.mean(ma[-100:])
    ok = ma >= tol * final
    for i in range(len(ok)):
        if ok[i:].mean() > 0.9:
            return i + n, final
    return len(rewards), final


def fig_cumulative(best, baselines):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        r = best[algo]["log"]["episode_rewards"]
        ax.plot(r, color=COLORS[algo], alpha=0.25, lw=0.7, label="reward per season")
        ma = moving_avg(r)
        ax.plot(np.arange(len(ma)) + 50, ma, color=COLORS[algo], lw=2, label="50-season average")
        ax.axhline(baselines["Rule-based seller"], color="grey", ls="--", lw=1, label="rule-based seller")
        ax.axhline(baselines["Always intermediary"], color="black", ls=":", lw=1, label="always intermediary")
        ax.set_title(f"{NAMES[algo]} (best run {best[algo]['run']})")
        ax.set_xlabel("Training season (episode)")
        ax.set_ylabel("Cumulative reward per season")
        ax.set_ylim(-12, 40)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Cumulative reward per season during training (best run of each method)")
    fig.tight_layout()
    fig.savefig(ASSETS / "cumulative_rewards.png")
    plt.close(fig)


def fig_stability(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for r in runs["dqn"]:
        f, v = zip(*r["curves"]["loss"])
        is_best = r["run"] == best["dqn"]["run"]
        ax.plot(f, v, color=COLORS["dqn"] if is_best else "#9bb8d3", lw=2 if is_best else 0.6, alpha=1 if is_best else 0.6,
                label=f"best run {r['run']}" if is_best else None)
    ax.set_yscale("log")
    ax.set_title("DQN objective: TD loss (all 10 runs, best highlighted)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("TD loss (log scale)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for algo in ["reinforce", "ppo", "a2c"]:
        f, v = zip(*best[algo]["log"]["curves"]["entropy"])
        ax.plot(f, v, color=COLORS[algo], lw=2, label=f"{NAMES[algo]} (run {best[algo]['run']})")
    ax.axhline(np.log(len(ACTION_NAMES)), color="grey", ls="--", lw=1, label="max entropy (uniform)")
    ax.set_title("Policy entropy of the policy-gradient methods")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_stability.png")
    plt.close(fig)


def fig_convergence(best):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for algo in ALGOS:
        r = best[algo]["log"]["episode_rewards"]
        ma = moving_avg(r)
        ep, final = convergence_episode(r)
        axes[0].plot(np.arange(len(ma)) + 50, ma, color=COLORS[algo], lw=2, label=NAMES[algo])
        axes[0].axvline(ep, color=COLORS[algo], ls=":", lw=1)
        rows.append(dict(method=NAMES[algo], run=best[algo]["run"], converged_at_season=ep,
                         final_train_reward=round(final, 2), total_seasons=len(r)))
    axes[0].set_title("50-season moving average (dotted = convergence point)")
    axes[0].set_xlabel("Training season")
    axes[0].set_ylabel("Reward per season")
    axes[0].legend(fontsize=8)
    df = pd.DataFrame(rows)
    axes[1].bar(df["method"], df["converged_at_season"], color=[COLORS[a] for a in ALGOS])
    for i, v in enumerate(df["converged_at_season"]):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Seasons needed to reach a stable level (95% of final)")
    axes[1].set_ylabel("Training seasons")
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png")
    plt.close(fig)
    df.to_csv(LOG_DIR / "convergence.csv", index=False)
    print("\nConvergence\n", df.to_string(index=False))


def fig_sweeps(runs, baselines):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        rr = sorted(runs[algo], key=lambda r: r["run"])
        vals = [r["evaluation"]["mean_reward"] for r in rr]
        errs = [r["evaluation"]["std_reward"] for r in rr]
        ax.bar([str(r["run"]) for r in rr], vals, yerr=errs, color=COLORS[algo], capsize=2)
        ax.axhline(baselines["Rule-based seller"], color="grey", ls="--", lw=1)
        ax.set_title(NAMES[algo])
        ax.set_xlabel("Run")
    axes[0].set_ylabel("Mean evaluation reward (20 unseen seasons)")
    fig.suptitle("Hyperparameter sweeps: evaluation reward of every run (dashed = rule-based seller)")
    fig.tight_layout()
    fig.savefig(ASSETS / "sweep_overview.png")
    plt.close(fig)


def generalisation(best):
    rows = []
    for scen in SCENARIOS:
        row = {"Scenario": scen}
        for algo in ALGOS:
            row[NAMES[algo]] = round(evaluate(best[algo]["net"], seeds=GEN_SEEDS, scenario=scen)["mean_reward"], 2)
        row["Rule-based"] = round(evaluate_policy(lambda o, e: rule_based_policy(e), GEN_SEEDS, scen)["mean_reward"], 2)
        row["Intermediary"] = round(evaluate_policy(lambda o, e: intermediary_policy(e), GEN_SEEDS, scen)["mean_reward"], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "generalization.csv", index=False)
    print("\nGeneralisation (mean reward per season, 20 unseen seasons per scenario)\n", df.to_string(index=False))
    cols = [NAMES[a] for a in ALGOS] + ["Rule-based", "Intermediary"]
    colors = [COLORS[a] for a in ALGOS] + ["#999999", "#444444"]
    x = np.arange(len(df))
    w = 0.13
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for k, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (k - 2.5) * w, df[c], w, label=c, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Scenario"])
    ax.set_ylabel("Mean reward per season")
    ax.set_title("Generalisation: best models on unseen seasons and changed market conditions")
    ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(ASSETS / "generalization.png")
    plt.close(fig)
    return df


@torch.no_grad()
def behaviour(best):
    rows = []
    for algo in ALGOS:
        c = collections.Counter()
        for s in EVAL_SEEDS:
            env = MarketLinkEnv()
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                a = int(best[algo]["net"](torch.as_tensor(obs)).argmax())
                c[a] += 1
                obs, r, t, tr, _ = env.step(a)
                done = t or tr
        total = sum(c.values())
        rows.append({"Method": NAMES[algo], **{ACTION_NAMES[i]: round(100 * c[i] / total, 1) for i in range(len(ACTION_NAMES))}})
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "behaviour_action_share.csv", index=False)
    print("\nShare of days each action is chosen (%), 20 unseen seasons\n", df.to_string(index=False))


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = {a: load_runs(a) for a in ALGOS}
    missing = [a for a in ALGOS if not runs[a]]
    if missing:
        raise SystemExit(f"No training logs for {missing}. Train them first (see README).")

    baselines = {
        "Always intermediary": evaluate_policy(lambda o, e: intermediary_policy(e)),
        "Rule-based seller": evaluate_policy(lambda o, e: rule_based_policy(e)),
        "Random actions": evaluate_policy(lambda o, e: e.action_space.sample()),
    }
    pd.DataFrame([{"Policy": k, **{kk: round(vv, 2) for kk, vv in v.items()}} for k, v in baselines.items()]) \
        .to_csv(LOG_DIR / "baselines.csv", index=False)
    base_reward = {k: v["mean_reward"] for k, v in baselines.items()}

    best = {}
    for algo in ALGOS:
        top = max(runs[algo], key=lambda r: r["evaluation"]["mean_reward"])
        path = MODEL_DIR / SUBDIR[algo] / f"{algo}_run_{top['run']:02d}.pt"
        net, _ = load_policy_net(path)
        best[algo] = dict(run=top["run"], log=top, eval=top["evaluation"], net=net, path=path)
    summary = {a: dict(run=b["run"], model=str(b["path"].relative_to(ROOT)).replace("\\", "/"),
                       evaluation=b["eval"], config=b["log"]["config"]) for a, b in best.items()}
    overall = max(summary, key=lambda a: summary[a]["evaluation"]["mean_reward"])
    (LOG_DIR / "best_models.json").write_text(json.dumps(dict(best_overall=overall, models=summary), indent=2))

    print("Baselines (20 unseen seasons):")
    for k, v in baselines.items():
        print(f"  {k:<20} reward {v['mean_reward']:6.2f}  waste {v['waste_pct']:4.1f}%  net {v['net_income_rwf'] / 1e6:.2f} M RWF")
    print("Best run per algorithm:")
    for a in ALGOS:
        e = summary[a]["evaluation"]
        print(f"  {NAMES[a]:<10} run {summary[a]['run']:>2}  reward {e['mean_reward']:6.2f} ± {e['std_reward']:.2f}"
              f"  waste {e['waste_pct']:4.1f}%  net {e['net_income_rwf'] / 1e6:.2f} M RWF")
    print(f"Best overall: {NAMES[overall]}")

    fig_cumulative(best, base_reward)
    fig_stability(best, runs)
    fig_convergence(best)
    fig_sweeps(runs, base_reward)
    generalisation(best)
    behaviour(best)
    print(f"\nFigures saved in {ASSETS}")


if __name__ == "__main__":
    main()
