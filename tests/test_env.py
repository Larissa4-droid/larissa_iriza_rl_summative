import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTION_NAMES, SEASON_DAYS, MarketLinkEnv, rule_based_policy


def test_gymnasium_api():
    check_env(MarketLinkEnv(), skip_render_check=True)


def test_spaces():
    env = MarketLinkEnv()
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == len(ACTION_NAMES) == 9
    assert obs.shape == (17,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_season_ends_after_45_days():
    env = MarketLinkEnv()
    env.reset(seed=1)
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(rule_based_policy(env))
        done = term or trunc
        steps += 1
    assert steps == SEASON_DAYS


def test_holding_everything_fails_the_season():
    env = MarketLinkEnv()
    env.reset(seed=2)
    done, steps = False, 0
    while not done:
        _, _, term, _, _ = env.step(0)
        done = term
        steps += 1
    assert steps < SEASON_DAYS and env.totals["wasted_kg"] > 0


def test_actions_change_state():
    env = MarketLinkEnv()
    env.reset(seed=3)
    for _ in range(3):
        env.step(0)
    before = env.total_stock
    env.step(5)                       # truck to Kigali
    assert env.truck_busy > 0
    assert env.totals["kigali_kg"] > 0 and env.total_stock < before + env.harvest_today
