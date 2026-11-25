"""
MarketLinkEnv: a selling advisor for a rural tomato cooperative in Rwanda
=========================================================================

Mission link
------------
Smallholder farmers in rural Rwanda often sell to intermediaries at the farm gate
for low prices because they lack market information and market access. The
market-linkage platform gives them price predictions, product listings, buyer
orders and mobile-money payments. This environment asks the next question:
*given that information, what should the cooperative actually do each day?*

The agent is the platform's selling advisor for one tomato cooperative during a
45-day harvest season. Every day it chooses one action: hold, sell to the
intermediary, sell at the local market, truck a load to Kigali, accept a buyer
order from the platform, put fresh stock into cold storage, or post a listing.

Why it is a sequential (RL) problem
-----------------------------------
* Tomatoes age every day and spoil after 5 days at ambient temperature, so
  holding stock for a better price later risks losing it.
* The truck needs a day to come back after a trip to Kigali, so shipping today
  removes that option tomorrow.
* A listing posted today only brings buyer orders over the next days.
* Cold storage costs money every day but slows ageing.
None of these can be solved by a one-off price prediction.

All numbers are simulation assumptions (documented below); prices are in RWF/kg.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Simulation parameters (assumptions)
# ----------------------------------------------------------------------------
SEASON_DAYS = 45
MAX_AGE = 5                     # ambient tomatoes spoil when they reach 5 days
QUALITY = np.array([1.00, 0.95, 0.85, 0.70, 0.50])   # price multiplier by age
COLD_CAPACITY = 1500.0          # kg of rented cold-room space
COLD_MOVE = 500.0               # kg moved into cold storage per action
COLD_COST = 5.0                 # RWF per kg per day
COLD_QUALITY = 0.92             # cold-stored tomatoes sell at 92 % of fresh price
COLD_DECAY = 0.01               # 1 % of cold stock is lost per day
LOCAL_LIMIT = 300.0             # kg the local rural market absorbs per day
TRUCK_CAPACITY = 1000.0         # kg per trip to Kigali (Nyabugogo)
TRUCK_FIXED = 60_000.0         # RWF per trip (hired truck + driver, district to Nyabugogo)
TRUCK_PER_KG = 20.0             # RWF per kg (loading, market fee)
LISTING_DAYS = 5                # a platform listing stays active 5 days
LISTING_FEE = 1_000.0           # RWF per listing
MOMO_FEE = 0.01                 # 1 % mobile-money fee on platform sales
ORDER_TTL = 2                   # buyer orders expire after 2 days
WASTE_LIMIT = 0.30              # season fails if >30 % of harvest is wasted (after day 10)
FAIL_PENALTY = 5.0
INVALID_PENALTY = 0.02
REWARD_UNIT = 100_000.0         # reward is measured in units of 100,000 RWF

ACTION_NAMES = [
    "HOLD",
    "INTERMEDIARY 25%",
    "INTERMEDIARY 50%",
    "LOCAL MARKET 25%",
    "LOCAL MARKET 50%",
    "TRUCK TO KIGALI",
    "ACCEPT ORDER",
    "COLD STORAGE",
    "POST LISTING",
]

OBS_NAMES = [
    "stock_age0", "stock_age1", "stock_age2", "stock_age3", "stock_age4",
    "cold_stock", "harvest_today", "price_farmgate", "price_local", "price_kigali",
    "forecast_kigali_3d", "best_order_qty", "best_order_price", "n_orders",
    "listing_days_left", "truck_available", "day_of_season",
]

SCENARIOS = {
    "normal": {},
    "price_glut": {"price_scale": 0.75},
    "big_harvest": {"harvest_scale": 1.5},
    "few_buyers": {"order_rate_scale": 0.35},
    "low_start_stock": {"start_stock": 0.0},
}


class MarketLinkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, scenario: str = "normal", **overrides):
        super().__init__()
        self.render_mode = render_mode
        self.base_options = {**SCENARIOS.get(scenario, {}), **overrides}
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0.0, 3.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self._renderer = None
        self.last_action = None
        self.last_event = ""

    # ------------------------------------------------------------------ setup
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        o = {**self.base_options, **(options or {})}
        rng = self.np_random
        self.harvest_scale = o.get("harvest_scale", 1.0)
        self.price_scale = o.get("price_scale", 1.0)
        self.order_rate_scale = o.get("order_rate_scale", 1.0)

        # Pre-generate the season's Kigali price path (mean-reverting, with a
        # mid-season glut when everyone harvests) and the harvest curve.
        t = np.arange(SEASON_DAYS + 4)
        mean = 430 - 140 * np.exp(-((t - 22) / 9.0) ** 2) + 2.0 * np.maximum(t - 30, 0)
        x, noise = 0.0, []
        for _ in t:
            x = 0.8 * x + rng.normal(0, 0.06)
            noise.append(x)
        self.kigali_path = self.price_scale * mean * np.exp(np.array(noise))
        h = 450 * np.exp(-((t - 20) / 11.0) ** 2) + 60
        self.harvest_path = self.harvest_scale * h * rng.lognormal(0, 0.2, size=len(t))

        self.day = 0
        self.stock = np.zeros(MAX_AGE)
        start = o.get("start_stock", rng.uniform(0, 300))
        self.stock[0] = start * 0.5
        self.stock[1] = start * 0.5
        self.cold = 0.0
        self.orders: list[dict] = []      # each: qty, price_factor, ttl
        self.listing_days = 0
        self.truck_busy = 0
        self.harvest_today = 0.0
        self.forecast = 0.0
        self._new_day()

        self.totals = dict(income=0.0, costs=0.0, wasted_kg=0.0, sold_kg=0.0, harvested_kg=float(start),
                           reward=0.0, intermediary_kg=0.0, local_kg=0.0, kigali_kg=0.0, platform_kg=0.0)
        self.last_action, self.last_event = None, "Season starts"
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def _new_day(self):
        """Prices, ML forecast and buyer orders for the current day."""
        rng = self.np_random
        k = self.kigali_path[self.day]
        self.price_kigali = k
        self.price_local = 0.75 * k * rng.uniform(0.95, 1.05)
        self.price_farmgate = 0.55 * k * rng.uniform(0.95, 1.05)
        # The platform's ML price model: true price in 3 days plus 8 % error.
        self.forecast = self.kigali_path[self.day + 3] * rng.normal(1.0, 0.08)
        rate = (0.7 if self.listing_days > 0 else 0.12) * self.order_rate_scale
        for _ in range(rng.poisson(rate)):
            self.orders.append(dict(qty=float(rng.uniform(100, 400)),
                                    price_factor=float(rng.uniform(0.85, 1.0)), ttl=ORDER_TTL))
        self.orders = sorted(self.orders, key=lambda o: -o["price_factor"])[:3]

    # ------------------------------------------------------------------ helpers
    def _take(self, kg: float, cold_ok: bool = True) -> float:
        """Remove up to kg from stock, oldest ambient first, then cold. Returns value
        multiplier-weighted kg (i.e. kg * quality) so the caller can price it."""
        need, value_kg = kg, 0.0
        for age in range(MAX_AGE - 1, -1, -1):
            q = min(self.stock[age], need)
            self.stock[age] -= q
            value_kg += q * QUALITY[age]
            need -= q
        if cold_ok and need > 0:
            q = min(self.cold, need)
            self.cold -= q
            value_kg += q * COLD_QUALITY
            need -= q
        return kg - need, value_kg

    @property
    def total_stock(self) -> float:
        return float(self.stock.sum() + self.cold)

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        action = int(action)
        income = costs = 0.0
        invalid = False
        event = ""
        stock = self.total_stock

        if action in (1, 2):                                      # intermediary
            if stock < 1:
                invalid = True
            else:
                kg, vkg = self._take(stock * (0.25 if action == 1 else 0.5))
                income = vkg * self.price_farmgate
                self.totals["intermediary_kg"] += kg
                self.totals["sold_kg"] += kg
                event = f"Sold {kg:.0f} kg to intermediary"
        elif action in (3, 4):                                    # local market
            if stock < 1:
                invalid = True
            else:
                kg, vkg = self._take(min(stock * (0.25 if action == 3 else 0.5), LOCAL_LIMIT))
                income = vkg * self.price_local
                self.totals["local_kg"] += kg
                self.totals["sold_kg"] += kg
                event = f"Sold {kg:.0f} kg at local market"
        elif action == 5:                                         # truck to Kigali
            if self.truck_busy > 0 or stock < 1:
                invalid = True
            else:
                kg, vkg = self._take(min(stock, TRUCK_CAPACITY))
                income = vkg * self.price_kigali
                costs = TRUCK_FIXED + TRUCK_PER_KG * kg
                self.truck_busy = 2                               # back in 1 day (counter ticks today)
                self.totals["kigali_kg"] += kg
                self.totals["sold_kg"] += kg
                event = f"Truck took {kg:.0f} kg to Kigali"
        elif action == 6:                                         # accept platform order
            if not self.orders or stock < 1:
                invalid = True
            else:
                o = self.orders.pop(0)
                kg, vkg = self._take(min(o["qty"], stock))
                income = vkg * self.price_kigali * o["price_factor"] * (1 - MOMO_FEE)
                self.totals["platform_kg"] += kg
                self.totals["sold_kg"] += kg
                event = f"Buyer order: {kg:.0f} kg, paid by mobile money"
        elif action == 7:                                         # cold storage
            space = COLD_CAPACITY - self.cold
            amb = self.stock[:2].sum()
            if space < 1 or amb < 1:
                invalid = True
            else:
                q = min(COLD_MOVE, space, amb)
                moved = 0.0
                for age in (0, 1):                                # freshest stock goes in
                    m = min(self.stock[age], q - moved)
                    self.stock[age] -= m
                    moved += m
                self.cold += moved
                event = f"Moved {moved:.0f} kg into cold storage"
        elif action == 8:                                         # post listing
            if self.listing_days > 0 or stock < 1:
                invalid = True
            else:
                self.listing_days = LISTING_DAYS
                costs = LISTING_FEE
                event = "Listing posted on the platform"
        else:
            event = "Held stock"

        if invalid:
            event = f"{ACTION_NAMES[action]} not possible"

        # ---- end of day: storage cost, ageing, spoilage, new harvest ------
        costs += COLD_COST * self.cold
        cold_loss = COLD_DECAY * self.cold
        self.cold -= cold_loss
        spoiled = self.stock[-1] + cold_loss
        self.stock[1:] = self.stock[:-1].copy()
        self.stock[0] = 0.0
        waste_value = spoiled * self.price_farmgate

        self.day += 1
        self.truck_busy = max(0, self.truck_busy - 1)
        self.listing_days = max(0, self.listing_days - 1)
        for o in self.orders:
            o["ttl"] -= 1
        self.orders = [o for o in self.orders if o["ttl"] > 0]

        terminated = False
        if self.day < SEASON_DAYS:
            self.harvest_today = float(self.harvest_path[self.day])
            self.stock[0] += self.harvest_today
            self.totals["harvested_kg"] += self.harvest_today
            self._new_day()

        reward = (income - costs - waste_value) / REWARD_UNIT - (INVALID_PENALTY if invalid else 0.0)

        self.totals["income"] += income
        self.totals["costs"] += costs
        self.totals["wasted_kg"] += spoiled

        waste_share = self.totals["wasted_kg"] / max(self.totals["harvested_kg"], 1.0)
        if self.day >= 10 and waste_share > WASTE_LIMIT:
            terminated = True
            reward -= FAIL_PENALTY
            event += " | season failed: too much waste"
        elif self.day >= SEASON_DAYS:
            terminated = True
            # leftover stock is sold to the intermediary at the end of the season
            left, vkg = self._take(self.total_stock)
            final = vkg * self.price_farmgate * 0.9
            self.totals["income"] += final
            reward += final / REWARD_UNIT
            event += f" | season over, {left:.0f} kg left sold at farm gate"

        self.totals["reward"] += reward
        self.last_action, self.last_event = action, event
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, False, self._info(invalid=invalid, spoiled=spoiled)

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        best = self.orders[0] if self.orders else None
        obs = np.array([
            *(self.stock / 1000.0),
            self.cold / COLD_CAPACITY,
            self.harvest_today / 600.0,
            self.price_farmgate / 600.0,
            self.price_local / 600.0,
            self.price_kigali / 600.0,
            self.forecast / 600.0,
            (best["qty"] / 400.0) if best else 0.0,
            (best["price_factor"] * self.price_kigali / 600.0) if best else 0.0,
            len(self.orders) / 3.0,
            self.listing_days / LISTING_DAYS,
            1.0 if self.truck_busy == 0 else 0.0,
            min(self.day, SEASON_DAYS) / SEASON_DAYS,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 3.0)

    def _info(self, **extra) -> dict:
        return dict(day=self.day, stock_kg=self.total_stock, cold_kg=self.cold,
                    income_rwf=self.totals["income"], costs_rwf=self.totals["costs"],
                    wasted_kg=self.totals["wasted_kg"], harvested_kg=self.totals["harvested_kg"],
                    net_rwf=self.totals["income"] - self.totals["costs"],
                    price_kigali=self.price_kigali, orders=len(self.orders), **extra)

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import MarketLinkRenderer
            self._renderer = MarketLinkRenderer(self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ----------------------------------------------------------------------------
# Simple baseline policies, used for comparison in the report
# ----------------------------------------------------------------------------
def intermediary_policy(env: MarketLinkEnv) -> int:
    """What many cooperatives do today: sell half of the stock at the farm gate every day."""
    return 2


def rule_based_policy(env: MarketLinkEnv) -> int:
    """A sensible hand-written rule using the platform's information."""
    if env.orders and env.total_stock > 50:
        return 6
    if env.listing_days == 0 and env.total_stock > 200:
        return 8
    if env.truck_busy == 0 and env.total_stock >= 700:
        return 5
    if env.stock[3:].sum() > 50:
        return 4
    return 0
