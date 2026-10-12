"""Bayesian player-talent model.

Each player has two underlying talents, estimated from their history with
older games counting for less:

  chance rate - expected goals (xG) they generate per game. A Gamma-Poisson
                estimate: their decayed xG total plus `a` games' worth of the
                average for their position, over their decayed games plus `a`.
  finishing   - goals scored per expected goal. Starts at 1 (average) with the
                strength of `c` expected goals, and moves towards the player's
                own goals / xG as their sample grows.

Expected goals tonight = chance rate x finishing x matchup, where the matchup
adjusts for the opponent's defence, their starting goalie and home ice.
P(goal) = 1 - exp(-expected goals), the chance of one or more Poisson goals.

The handful of parameters are fitted by maximum likelihood on one season,
choosing between the half-lives precomputed in tims_features.
"""
import itertools

import numpy as np
from scipy.optimize import minimize

from tims_features import BAYES_FINISH_HL, BAYES_RATE_HL


class BayesTalent:
    PARAMS = ["log_a", "log_c", "log_mu_f", "log_mu_d", "b0", "b_opp", "b_goalie", "b_home"]

    def _parts(self, df, theta, hl_rate, hl_fin):
        p = dict(zip(self.PARAMS, theta))
        a, c = np.exp(p["log_a"]), np.exp(p["log_c"])
        mu = np.where(df["is_defense"] == 1, np.exp(p["log_mu_d"]), np.exp(p["log_mu_f"]))
        n = df[f"b_n_{hl_rate}"].fillna(0).values
        x = df[f"b_x_{hl_rate}"].fillna(0).values
        rate = (x + a * mu) / (n + a)
        g = df[f"b_g_{hl_fin}"].fillna(0).values
        xf = df[f"b_xf_{hl_fin}"].fillna(0).values
        finishing = (g + c) / (xf + c)
        opp = np.log((df["opp_xga_l20"] / self.league_xga).clip(0.5, 2).fillna(1)).values
        matchup = np.exp(p["b0"] + p["b_opp"] * opp + p["b_goalie"] * df["goalie_gsax_l40"].fillna(0).values
                         + p["b_home"] * df["home"].values)
        return rate * finishing * matchup

    def _nll(self, theta, df, y, hl_rate, hl_fin):
        lam = self._parts(df, theta, hl_rate, hl_fin)
        p = np.clip(1 - np.exp(-lam), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, df):
        """Fit on feature rows (from tims_features) that include `scored`."""
        self.league_xga = df["opp_xga_l20"].mean()
        y = df["scored"].values
        start = np.array([np.log(10), np.log(30), np.log(0.2), np.log(0.06), 0.0, 1.0, -1.0, 0.0])
        best = None
        for hl_rate, hl_fin in itertools.product(BAYES_RATE_HL, BAYES_FINISH_HL):
            res = minimize(self._nll, start, args=(df, y, hl_rate, hl_fin), method="L-BFGS-B")
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x, hl_rate, hl_fin)
        self.loss, self.theta, self.hl_rate, self.hl_fin = best
        return self

    def predict(self, df):
        """P(goal) for each row."""
        return 1 - np.exp(-self._parts(df, self.theta, self.hl_rate, self.hl_fin))

    def describe(self):
        p = dict(zip(self.PARAMS, self.theta))
        return (f"chance-rate half-life {self.hl_rate} games, prior {np.exp(p['log_a']):.1f} games "
                f"(F {np.exp(p['log_mu_f']):.3f} / D {np.exp(p['log_mu_d']):.3f} xG per game); "
                f"finishing half-life {self.hl_fin} games, prior {np.exp(p['log_c']):.0f} xG; "
                f"opponent {p['b_opp']:+.2f}, goalie {p['b_goalie']:+.2f}, home {p['b_home']:+.2f}")
