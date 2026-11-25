"""2D visualisation of MarketLinkEnv with pygame.

Layout
------
* Left:   the cooperative's storehouse (5 crates coloured by tomato age) and the cold room.
* Centre: the three selling channels (farm gate, local market, Kigali) with today's prices,
          and the truck on the road to Kigali.
* Right:  a phone showing the market-linkage platform (listing, buyer orders, ML forecast).
* Bottom: season dashboard (income, costs, waste, net) and the Kigali price chart.
"""
from __future__ import annotations

import numpy as np
import pygame

W, H = 1180, 720
BG = (246, 241, 230)
INK = (34, 38, 46)
MUTED = (110, 112, 120)
CARD = (255, 255, 255)
LINE = (214, 206, 190)
GREEN = (46, 140, 87)
RED = (205, 60, 50)
ORANGE = (230, 140, 40)
BLUE = (40, 110, 190)
SOIL = (120, 84, 54)
AGE_COLORS = [(214, 40, 40), (225, 75, 45), (210, 120, 50), (165, 130, 60), (110, 95, 70)]
AGE_LABELS = ["fresh", "1 day", "2 days", "3 days", "4 days (spoils tomorrow)"]


class MarketLinkRenderer:
    def __init__(self, mode: str, fps: int):
        self.mode = mode
        self.fps = fps
        pygame.init()
        if mode == "human":
            pygame.display.set_caption("MarketLink RL: tomato cooperative selling advisor")
            self.screen = pygame.display.set_mode((W, H))
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.Font(None, 34)
        self.f_big = pygame.font.Font(None, 30)
        self.f = pygame.font.Font(None, 22)
        self.f_small = pygame.font.Font(None, 18)
        self.price_hist: list[float] = []
        self.last_day = -1

    # ------------------------------------------------------------------ utils
    def text(self, s, pos, font=None, color=INK, center=False):
        img = (font or self.f).render(str(s), True, color)
        r = img.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        self.screen.blit(img, r)
        return r

    def card(self, rect, title=None, accent=None):
        pygame.draw.rect(self.screen, CARD, rect, border_radius=12)
        pygame.draw.rect(self.screen, accent or LINE, rect, 2, border_radius=12)
        if title:
            self.text(title, (rect[0] + 14, rect[1] + 10), self.f_big)

    # ------------------------------------------------------------------ draw
    def draw(self, env):
        if self.mode == "human":
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
        if env.day == 0 or env.day < self.last_day:
            self.price_hist = []
        if env.day != self.last_day:
            self.price_hist.append(env.price_kigali)
            self.last_day = env.day

        s = self.screen
        s.fill(BG)
        # header + season progress
        self.text("MarketLink: selling advisor for a tomato cooperative", (24, 16), self.f_title)
        self.text(f"Day {min(env.day + 1, 45)} of 45", (W - 190, 18), self.f_title, GREEN)
        pygame.draw.rect(s, LINE, (24, 52, W - 48, 8), border_radius=4)
        pygame.draw.rect(s, GREEN, (24, 52, int((W - 48) * min(env.day, 45) / 45), 8), border_radius=4)

        self._storehouse(env)
        self._markets(env)
        self._phone(env)
        self._dashboard(env)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def _storehouse(self, env):
        s = self.screen
        x, y, w, h = 24, 76, 330, 500
        self.card((x, y, w, h), "Cooperative storehouse")
        self.text(f"Harvest today: {env.harvest_today:,.0f} kg", (x + 14, y + 40), self.f, MUTED)
        # crates by age
        cx, cy = x + 20, y + 72
        scale = 160 / 800.0
        for age in range(5):
            kg = env.stock[age]
            bar_w = int(min(kg * scale, 160))
            pygame.draw.rect(s, (238, 232, 220), (cx + 120, cy + age * 40, 170, 26), border_radius=6)
            pygame.draw.rect(s, AGE_COLORS[age], (cx + 120, cy + age * 40, max(bar_w, 2), 26), border_radius=6)
            self.text(AGE_LABELS[age].split(" (")[0], (cx, cy + age * 40 + 5), self.f)
            self.text(f"{kg:,.0f} kg", (cx + 126, cy + age * 40 + 6), self.f_small, (255, 255, 255) if bar_w > 60 else INK)
        self.text("Red = fresh  ->  brown = about to spoil", (cx, cy + 202), self.f_small, MUTED)
        self.text(f"In store: {env.total_stock:,.0f} kg", (cx, cy + 228), self.f_big, INK)
        # cold room
        ry = y + 360
        pygame.draw.rect(s, (225, 238, 250), (x + 14, ry, w - 28, 100), border_radius=10)
        pygame.draw.rect(s, BLUE, (x + 14, ry, w - 28, 100), 2, border_radius=10)
        self.text("Cold room (rented)", (x + 28, ry + 10), self.f, BLUE)
        fill = env.cold / 1500.0
        pygame.draw.rect(s, (255, 255, 255), (x + 28, ry + 40, w - 56, 22), border_radius=6)
        pygame.draw.rect(s, BLUE, (x + 28, ry + 40, int((w - 56) * fill), 22), border_radius=6)
        self.text(f"{env.cold:,.0f} / 1,500 kg   cost 5 RWF/kg/day", (x + 28, ry + 70), self.f_small, INK)

    def _markets(self, env):
        s = self.screen
        x, y = 372, 76
        a = env.last_action
        channels = [
            ("Farm gate", "intermediary", env.price_farmgate, (1, 2), SOIL),
            ("Local market", "max 300 kg/day", env.price_local, (3, 4), ORANGE),
            ("Kigali (Nyabugogo)", "truck 1,000 kg", env.price_kigali, (5,), GREEN),
        ]
        for i, (name, note, price, acts, col) in enumerate(channels):
            rx, ry = x, y + i * 112
            active = a in acts
            self.card((rx, ry, 410, 100), None, col if active else None)
            pygame.draw.rect(self.screen, col, (rx, ry, 10, 100), border_top_left_radius=12, border_bottom_left_radius=12)
            self.text(name, (rx + 24, ry + 14), self.f_big)
            self.text(note, (rx + 24, ry + 46), self.f_small, MUTED)
            self.text(f"{price:,.0f} RWF/kg", (rx + 250, ry + 16), self.f_big, col)
            if active:
                self.text("SELLING TODAY", (rx + 250, ry + 52), self.f, col)
        # truck road
        ry = y + 340
        self.card((x, ry, 410, 80))
        pygame.draw.line(s, (170, 160, 140), (x + 20, ry + 50), (x + 390, ry + 50), 4)
        busy = env.truck_busy > 0
        tx = x + 300 if busy else x + 40
        pygame.draw.rect(s, GREEN if not busy else MUTED, (tx, ry + 22, 56, 24), border_radius=4)
        pygame.draw.rect(s, GREEN if not busy else MUTED, (tx + 56, ry + 30, 18, 16), border_radius=3)
        for wx in (tx + 12, tx + 50):
            pygame.draw.circle(s, INK, (wx, ry + 48), 6)
        self.text("Truck at cooperative, ready" if not busy else "Truck on the way back from Kigali",
                  (x + 20, ry + 8), self.f_small, MUTED)
        # last decision
        from environment.custom_env import ACTION_NAMES
        self.card((x, 504, 410, 72), None, GREEN)
        self.text("Agent decision", (x + 16, 512), self.f_small, MUTED)
        self.text(ACTION_NAMES[a] if a is not None else "-", (x + 16, 530), self.f_big, GREEN)
        self.text(env.last_event[:58], (x + 16, 556), self.f_small, INK)

    def _phone(self, env):
        s = self.screen
        x, y, w, h = 800, 76, 356, 500
        pygame.draw.rect(s, INK, (x, y, w, h), border_radius=26)
        pygame.draw.rect(s, (250, 250, 252), (x + 10, y + 14, w - 20, h - 28), border_radius=18)
        px = x + 26
        self.text("MarketLink app", (px, y + 30), self.f_big, GREEN)
        # ML forecast
        pygame.draw.rect(s, (232, 245, 236), (px - 4, y + 64, w - 44, 70), border_radius=10)
        self.text("ML price forecast, Kigali in 3 days", (px + 6, y + 72), self.f_small, MUTED)
        arrow = "up" if env.forecast > env.price_kigali else "down"
        self.text(f"{env.forecast:,.0f} RWF/kg ({arrow})", (px + 6, y + 96), self.f_big,
                  GREEN if arrow == "up" else RED)
        # listing
        live = env.listing_days > 0
        pygame.draw.rect(s, (255, 244, 226) if live else (240, 240, 240), (px - 4, y + 146, w - 44, 50), border_radius=10)
        self.text("Listing: LIVE" if live else "Listing: none", (px + 6, y + 152), self.f, ORANGE if live else MUTED)
        self.text(f"{env.listing_days} day(s) left" if live else "post one to attract buyers",
                  (px + 6, y + 174), self.f_small, MUTED)
        # orders
        self.text("Buyer orders", (px, y + 210), self.f, INK)
        if not env.orders:
            self.text("No orders waiting", (px, y + 236), self.f_small, MUTED)
        for i, o in enumerate(env.orders):
            oy = y + 236 + i * 70
            pygame.draw.rect(s, CARD, (px - 4, oy, w - 44, 60), border_radius=10)
            pygame.draw.rect(s, BLUE, (px - 4, oy, w - 44, 60), 2, border_radius=10)
            self.text(f"{o['qty']:,.0f} kg at {o['price_factor'] * env.price_kigali:,.0f} RWF/kg", (px + 6, oy + 8), self.f, INK)
            self.text(f"Paid by mobile money - expires in {o['ttl']} day(s)", (px + 6, oy + 34), self.f_small, MUTED)

    def _dashboard(self, env):
        s = self.screen
        t = env.totals
        y = 590
        stats = [
            ("Income", f"{t['income'] / 1e6:,.2f} M RWF", GREEN),
            ("Costs", f"{t['costs'] / 1e3:,.0f} k RWF", ORANGE),
            ("Wasted", f"{t['wasted_kg']:,.0f} kg", RED),
            ("Sold", f"{t['sold_kg']:,.0f} kg", INK),
            ("Reward", f"{t['reward']:+.2f}", BLUE),
        ]
        for i, (k, v, col) in enumerate(stats):
            rx = 24 + i * 152
            self.card((rx, y, 142, 110))
            self.text(k, (rx + 12, y + 14), self.f, MUTED)
            self.text(v, (rx + 12, y + 50), self.f_big, col)
        # price chart
        cx, cy, cw, ch = 800, y, 356, 110
        self.card((cx, cy, cw, ch))
        self.text("Kigali price (RWF/kg)", (cx + 12, cy + 8), self.f_small, MUTED)
        hist = list(env.kigali_path[: min(env.day, 44) + 1])
        if len(hist) > 1:
            lo, hi = 150, 650
            pts = [(cx + 14 + i * (cw - 28) / 44, cy + ch - 12 - (p - lo) / (hi - lo) * (ch - 36)) for i, p in enumerate(hist)]
            pygame.draw.lines(s, GREEN, False, pts, 3)
            fx = pts[-1][0] + 3 * (cw - 28) / 44
            fy = cy + ch - 12 - (env.forecast - lo) / (hi - lo) * (ch - 36)
            pygame.draw.circle(s, ORANGE, (int(min(fx, cx + cw - 8)), int(fy)), 5)
            self.text("o forecast", (cx + cw - 90, cy + 8), self.f_small, ORANGE)

    def close(self):
        pygame.display.quit()
        pygame.quit()
