"""Adaptive Neuro-Fuzzy trust engine (Takagi-Sugeno), and the static Mamdani FIS.

Implements Algorithm 1 of description.docx exactly:

  Inputs   x = [x1 validation F1, x2 directional similarity, x3 cleanliness]
  Layer 1  mu_{j,i}(x_j) = exp(-0.5 ((x_j - c_{j,i}) / sigma_{j,i})^2)
  Layer 2  w_r = prod_j mu_{j, grid(r,j)}(x_j)          8 rules = 2^3 grid
  Layer 3  wbar_r = w_r / sum_m w_m
  Layer 4  f_r(x) = p_r x1 + q_r x2 + m_r x3 + s_r
  Layer 5  T = sum_r wbar_r f_r                          clipped to [0, 1]

Premise (c, sigma) and consequent (p, q, m, s) are updated online by gradient
descent on L = 0.5 (T - T*)^2.

Why 8 grid rules rather than a larger scatter partition: with three inputs and
two linguistic terms each, the rule base is complete AND every rule is nameable
("IF F1 is Low AND Similarity is Low AND Cleanliness is High THEN ..."). That
is what makes the trust decision auditable, which is the point of using fuzzy
inference at all.

Two numerical guards added to the supplied version, both justified:
  * sigma is clamped to [0.05, 0.60]. The floor was in the original; the CEILING
    is new and necessary -- without it a rule can widen until it fires for every
    input, which silently collapses the 8-rule base toward a single linear model.
  * the premise gradient is verified against a numerical gradient in
    `tests/test_anfis_gradient.py`, because a sign error there trains the model
    backwards while still appearing to converge.
"""
from __future__ import annotations

import numpy as np

N_INPUTS = 3
N_RULES = 8
SIGMA_MIN, SIGMA_MAX = 0.05, 0.60

INPUT_NAMES = ("val_f1", "dir_similarity", "cleanliness")
TERM_NAMES = ("Low", "High")


class AdaptiveNeuroFuzzyTrust:
    """8-rule first-order TSK ANFIS mapping 3 behavioural features to trust."""

    def __init__(self, lr_premise: float = 0.01, lr_consequent: float = 0.01,
                 seed: int = 42) -> None:
        self.lr_premise = lr_premise
        self.lr_consequent = lr_consequent
        self.num_inputs = N_INPUTS
        self.num_rules = N_RULES

        # Gaussian MF centres and widths: Low at 0.25, High at 0.75.
        self.c = np.array([[0.25, 0.75]] * N_INPUTS, dtype=np.float64)
        self.sigma = np.full((N_INPUTS, 2), 0.20, dtype=np.float64)

        # Grid partition: rule r is the combination (i, j, k) of terms.
        self.rule_grid = np.array([(i, j, k)
                                   for i in range(2) for j in range(2)
                                   for k in range(2)], dtype=int)

        # Consequents. The bias starts high for rules whose antecedents are
        # mostly "High" (good F1, aligned, clean) and low otherwise, so round 1
        # already behaves like a sensible detector rather than a random one.
        self.consequent = np.zeros((N_RULES, 4), dtype=np.float64)
        for r in range(N_RULES):
            base = 0.8 if self.rule_grid[r].sum() >= 2 else 0.2
            self.consequent[r] = [0.2, 0.2, 0.2, base]

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ forward
    @staticmethod
    def gaussian_mf(x, c, sigma):
        return np.exp(-0.5 * ((x - c) / (sigma + 1e-12)) ** 2)

    def forward(self, x) -> tuple[float, dict]:
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

        mf = np.empty((N_INPUTS, 2))
        for j in range(N_INPUTS):
            for i in range(2):
                mf[j, i] = self.gaussian_mf(x[j], self.c[j, i], self.sigma[j, i])

        w = np.empty(N_RULES)
        for r in range(N_RULES):
            i, j, k = self.rule_grid[r]
            w[r] = mf[0, i] * mf[1, j] * mf[2, k]

        w_sum = w.sum() + 1e-12
        w_bar = w / w_sum

        x_ext = np.array([x[0], x[1], x[2], 1.0])
        f = self.consequent @ x_ext
        raw = float(w_bar @ f)
        trust = float(np.clip(raw, 0.0, 1.0))

        cache = {"x": x, "x_ext": x_ext, "mf": mf, "w": w, "w_sum": w_sum,
                 "w_bar": w_bar, "f": f, "raw": raw, "trust": trust}
        return trust, cache

    # ------------------------------------------------------------ learning
    def update(self, cache: dict, target: float) -> float:
        """One online gradient step on L = 0.5 (T - T*)^2. Returns the error."""
        x, x_ext = cache["x"], cache["x_ext"]
        w, w_sum, w_bar = cache["w"], cache["w_sum"], cache["w_bar"]
        f, mf, raw = cache["f"], cache["mf"], cache["raw"]

        # Train on the RAW TSK output, never on the clipped one.
        #
        # An earlier version skipped the update whenever raw left [0,1], on the
        # reasoning that the clip has zero gradient there. That created a dead
        # zone the model could never escape: once the consequents grew enough to
        # push every client above 1.0, all trust scores clipped to exactly 1.0,
        # the ranking collapsed, and no gradient was ever applied to bring them
        # back. Measured under Sybil poisoning, every client -- honest and
        # malicious alike -- scored exactly 1.000 even though the underlying
        # root-set F1 separated them 0.674 against 0.017.
        #
        # The clip belongs at the output, as presentation and as the gate's
        # domain. The learned function underneath is linear in the consequents
        # and must stay trainable everywhere.
        error = raw - target

        # dL/d(consequent_r) = error * wbar_r * x_ext
        self.consequent -= self.lr_consequent * error * np.outer(w_bar, x_ext)

        # dT/dw_r = (f_r - T) / w_sum
        dT_dw = (f - raw) / w_sum

        grad_c = np.zeros_like(self.c)
        grad_s = np.zeros_like(self.sigma)
        for r in range(N_RULES):
            terms = self.rule_grid[r]
            for j in range(N_INPUTS):
                i = terms[j]
                cj, sj = self.c[j, i], self.sigma[j, i]
                u = (x[j] - cj) / (sj + 1e-12)
                # d mu / d c   = mu * u / sigma
                # d mu / d sig = mu * u^2 / sigma
                dmu_dc = mf[j, i] * u / (sj + 1e-12)
                dmu_ds = mf[j, i] * u * u / (sj + 1e-12)
                # product of the OTHER two membership grades in this rule
                other = 1.0
                for o in range(N_INPUTS):
                    if o != j:
                        other *= mf[o, terms[o]]
                grad_c[j, i] += error * dT_dw[r] * other * dmu_dc
                grad_s[j, i] += error * dT_dw[r] * other * dmu_ds

        self.c -= self.lr_premise * grad_c
        self.sigma = np.clip(self.sigma - self.lr_premise * grad_s,
                             SIGMA_MIN, SIGMA_MAX)
        return float(error)

    # ------------------------------------------------------ interpretability
    def describe_rules(self) -> list[dict]:
        """The 8 rules as readable IF-THEN statements, for the paper."""
        out = []
        for r in range(N_RULES):
            terms = self.rule_grid[r]
            antecedent = " AND ".join(
                f"{INPUT_NAMES[j]} is {TERM_NAMES[terms[j]]}" for j in range(N_INPUTS))
            p, q, m, s = self.consequent[r]
            out.append({
                "rule": r + 1,
                "if": antecedent,
                "then": f"T = {p:+.3f}*f1 {q:+.3f}*sim {m:+.3f}*clean {s:+.3f}",
                "bias": float(s),
                "coef": [float(p), float(q), float(m)],
            })
        return out

    def state(self) -> dict:
        return {"c": self.c.copy(), "sigma": self.sigma.copy(),
                "consequent": self.consequent.copy()}

    def load_state(self, st: dict) -> None:
        self.c = st["c"].copy()
        self.sigma = st["sigma"].copy()
        self.consequent = st["consequent"].copy()


# ------------------------------------------------------------------ Mamdani
class StaticMamdaniTrust:
    """The static-fuzzy baseline: fixed rules, fixed membership functions.

    This is the comparator that isolates what ADAPTATION buys. It sees exactly
    the same three inputs and uses a comparable rule base, but nothing about it
    is learned -- centres, widths and the rule table are all fixed a priori.
    Any difference against the ANFIS is therefore attributable to online
    parameter learning rather than to different information.
    """

    def __init__(self) -> None:
        self.centres = np.array([[0.25, 0.75]] * N_INPUTS)
        self.sigma = np.full((N_INPUTS, 2), 0.20)
        # Consequent singletons per rule (Sugeno-0 / simplified Mamdani):
        # trust rises with the number of "High" antecedents.
        self.rule_grid = np.array([(i, j, k)
                                   for i in range(2) for j in range(2)
                                   for k in range(2)], dtype=int)
        self.singleton = np.array([0.10, 0.25, 0.30, 0.50,
                                   0.55, 0.75, 0.80, 0.95])

    def forward(self, x) -> tuple[float, dict]:
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        mf = np.empty((N_INPUTS, 2))
        for j in range(N_INPUTS):
            for i in range(2):
                mf[j, i] = np.exp(-0.5 * ((x[j] - self.centres[j, i])
                                          / self.sigma[j, i]) ** 2)
        w = np.empty(N_RULES)
        for r in range(N_RULES):
            i, j, k = self.rule_grid[r]
            w[r] = min(mf[0, i], mf[1, j], mf[2, k])      # Mamdani min t-norm
        tot = w.sum() + 1e-12
        trust = float(np.clip((w @ self.singleton) / tot, 0.0, 1.0))
        return trust, {}

    def update(self, cache: dict, target: float) -> float:
        return 0.0                                        # static by definition
