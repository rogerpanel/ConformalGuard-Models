"""Online adaptive conformal prediction (Gibbs & Candès, 2021).

The protocol updates the threshold τ at each step:

    τ_{t+1} = τ_t + γ · ( 1{y_t ∉ C_{α_t}(G_<t)} − α )

Equivalently, the *effective* α is updated rather than τ:

    α_{t+1} = α_t + γ · ( err_t − α )

Theorem 2 (Sec. 4.4) proves that, regardless of distribution shift,

    lim sup_T   |1/T Σ_{t≤T} 1{err_t} − α|  →  0.

Hyper-parameters from the paper:
    γ = 0.05  (online learning rate),
    α = 0.05.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GibbsCandesState:
    alpha: float
    target_alpha: float
    gamma: float
    step: int
    cumulative_err: int


class GibbsCandesController:
    """Stateful online adaptive controller.

    Use as
        ctl = GibbsCandesController(target_alpha=0.05, gamma=0.05)
        for t in stream:
            tau_t = calibrator.threshold()  # base τ
            tau_eff = tau_t + ctl.adjustment(tau_t)
            err = int(true_label not in prediction_set)
            ctl.update(err)
    """

    def __init__(self, target_alpha: float = 0.05, gamma: float = 0.05,
                 alpha_min: float = 1e-3, alpha_max: float = 0.5):
        if not 0 < target_alpha < 1:
            raise ValueError("target_alpha must be in (0, 1)")
        self.target = target_alpha
        self.gamma = gamma
        self.alpha = target_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._cum_err = 0
        self._step = 0

    # -- update -----------------------------------------------------------

    def update(self, error_indicator: int) -> float:
        """`error_indicator = 1{y_t ∉ C}`. Returns the new α."""
        self.alpha = self.alpha + self.gamma * (error_indicator - self.target)
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))
        self._cum_err += int(error_indicator)
        self._step += 1
        return self.alpha

    # -- threshold adjustment --------------------------------------------

    def adjustment(self, base_threshold: float) -> float:
        """Heuristic mapping from α-drift back to a τ-shift.

        We use the inverse-quantile linearization Δτ ≈ (α* − α) · base_threshold.
        For a more principled mapping the user can pass a calibration ECDF.
        """
        return (self.target - self.alpha) * abs(base_threshold)

    # -- diagnostics ------------------------------------------------------

    def empirical_miscoverage(self) -> float:
        return self._cum_err / max(1, self._step)

    @property
    def state(self) -> GibbsCandesState:
        return GibbsCandesState(
            alpha=self.alpha,
            target_alpha=self.target,
            gamma=self.gamma,
            step=self._step,
            cumulative_err=self._cum_err,
        )
