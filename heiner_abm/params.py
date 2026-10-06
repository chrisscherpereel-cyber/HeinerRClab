"""Parameter containers for the Heiner cobweb agent-based model.

Baseline calibration:
    max price 100, min price 10, quantity range 1500, initial raw-material cost 45,
    max raw-material cost 80, max cost change 10, initial production 200,
    min production 15, flexibility slope 0.25, intercept 0, desired margin 5,
    threshold 25, four firms, 25 burn-in periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import List, Optional

# Production rules. "Bertrand" is the configuration value of the margin-feedback quantity rule (kept for compatibility
# with saved scenarios and registered results); it is not price-setting Bertrand competition. See heiner_abm.terminology.
RULE_TYPES = ("Bertrand", "Cournot")
SELECTION_RULES = ("Always", "Never", "Small", "Large", "Adaptive")
# Feedback the Adaptive selection rule learns from. Either way, feedback about a decision is released only once every
# period it covers has occurred (see DECISION_SCHEDULE in heiner_abm/agents.py):
#   "observable" - (default) the agent judges its decision from what it has observed over the `window` periods
#                  starting with the decision: its own and its rivals' realized output, realized costs and prices on its
#                  *believed* demand curve.
#   "oracle"     - ORACLE TREATMENT, researcher-only: the researcher's forked-market counterfactual over the measurement
#                  horizon H (true demand curve, rivals' simulated reactions), released at period t + H - 1. It is
#                  never used unless chosen explicitly, and results computed with it must be labeled as oracle results.
# The former "lookahead" option released the H-period counterfactual immediately, before the periods it covers had
# occurred. It was removed on 6 October 2026; with H = 1 it is identical to "oracle".
ADAPTIVE_FEEDBACK = ("observable", "oracle")

SELECTION_HELP = {
    "Always": "Always adopt the production rule's recommendation (maximally flexible).",
    "Never": "Never deviate from the default rule B: keep last period's production (fully rigid).",
    "Small": "SR1 - deviate only when the recommended change |q* - q| < threshold (change is risky).",
    "Large": "SR2 - deviate only when |q* - q| > threshold (large signals are clear signals).",
    "Adaptive": "Learns, per size-of-change bin, whether deviating has paid off and deviates only where its learned "
                "net gain is positive. It judges each past deviation, once the periods it covers have happened, from "
                "realized outcomes priced on its own believed demand curve (no future information).",
}


@dataclass
class MarketParams:
    p_max: float = 100.0          # demand intercept (never reached because q >= q_min)
    p_min: float = 10.0           # price floor; also the lower reflecting bound for cost
    q_range: float = 1500.0       # quantity at which demand hits the floor -> slope
    c0: float = 45.0              # initial raw-material cost
    c_max: float = 80.0           # upper reflecting bound for raw-material cost
    delta: float = 10.0           # max per-period raw-material cost change (volatility)

    @property
    def slope(self) -> float:
        return (self.p_max - self.p_min) / self.q_range

    def validate(self) -> List[str]:
        errs = []
        if self.p_max <= self.p_min:
            errs.append("Max price must exceed min price.")
        if self.c_max <= self.p_min:
            errs.append("Max raw-material cost must exceed min price (the lower cost bound).")
        if self.delta > self.c_max - self.p_min:
            errs.append("Cost volatility (max change) cannot exceed Max cost - Min price.")
        if not (self.p_min <= self.c0 <= self.c_max):
            errs.append("Initial cost must lie between Min price and Max cost.")
        if self.q_range <= 0:
            errs.append("Quantity range must be positive.")
        return errs


@dataclass
class FirmSpec:
    rule: str = "Bertrand"        # production rule for q*: "Bertrand" = margin-feedback quantity rule, or "Cournot"
    flex: float = 0.25            # phi: Cournot weight (0..1) or margin-feedback sensitivity
    selection: str = "Always"     # selection rule for accepting q* vs default rule B (hold q)
    threshold: float = 25.0       # threshold for Small/Large selection rules
    desired_margin: float = 5.0   # m*: desired margin of the margin-feedback rule (competition intensity)
    foresight: float = 0.0        # kappa: competence - fraction of next cost change anticipated
    noise: float = 0.0            # sigma: perception noise on the cost estimate
    label: Optional[str] = None


@dataclass
class GlobalFirmParams:
    q0: float = 200.0             # initial production of every firm
    q_min: float = 15.0           # minimum production
    flex_cost_slope: float = 0.0  # a: per-period flexibility cost = a*phi + b
    fixed_cost: float = 0.0       # b: per-period fixed cost
    margin_includes_fixed: bool = False  # margin-feedback rule uses P - c - F/q (on) vs P - c (off)


@dataclass
class AdaptiveParams:
    """Learning of the Adaptive selection rule. With feedback "observable" (default), the decision made at period d is
    judged at the end of period d + window - 1: the agent compares holding the recommended level with holding its old
    level (rule B) over those `window` periods, with rivals' actual output, realized costs and prices on its believed
    demand curve, all of which it has observed by then. With feedback "oracle" it instead receives the researcher's
    forked-market counterfactual over the measurement horizon H, at the end of period d + H - 1 (see ADAPTIVE_FEEDBACK).
    Feedback whose horizon would end after the last simulated period is never released: no partial feedback."""
    bin_edges: tuple = (5.0, 15.0, 30.0, 60.0)   # |q* - q| bin boundaries
    memory: float = 0.97                          # exponential forgetting (lambda)
    feedback: str = "observable"                  # "observable" or "oracle" (researcher-only treatment)
    window: int = 20                              # periods over which an observable judgement is made (>= 1)


@dataclass
class EvolutionParams:
    enabled: bool = False
    every: int = 50               # periods between flexibility revisions
    imitation_prob: float = 0.5   # prob. a firm copies the most profitable firm's phi
    mutation_sd: float = 0.03     # experimentation noise on phi
    flex_min: float = 0.0
    flex_max: float = 1.0


@dataclass
class StructuralParams:
    """Unannounced structural change: regime shifts in the demand curve that firms are not told about.

    With probability `hazard` per period a new demand regime is drawn around the baseline:
        p_max = base p_max + intercept_sd * z1        (clipped to >= p_min + 10)
        slope = base slope * exp(slope_sd * z2)        (clipped to [0.25, 4] x base)
    Regimes persist until the next shift. Model-based (Cournot) firms compute best replies with
    *believed* demand parameters: the regime of `belief_lag` periods before their information set
    (belief_lag = 0: they learn a regime one period after it starts; belief_lag < 0: they never
    update and keep the baseline model). Model-free (margin-feedback) firms only use observed prices. The shift
    process is specified here, so it is unknown to the firms but not unknowable (see terminology.KNIGHT_NOTE).
    """
    enabled: bool = False
    hazard: float = 0.02
    intercept_sd: float = 10.0
    slope_sd: float = 0.3
    belief_lag: int = 20


@dataclass
class Scenario:
    market: MarketParams = field(default_factory=MarketParams)
    firms: List[FirmSpec] = field(default_factory=list)
    firm_globals: GlobalFirmParams = field(default_factory=GlobalFirmParams)
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)
    evolution: EvolutionParams = field(default_factory=EvolutionParams)
    structural: StructuralParams = field(default_factory=StructuralParams)
    periods: int = 1000
    burn_in: int = 25
    seed: int = 0

    @property
    def n_firms(self) -> int:
        return len(self.firms)

    def flex_cost(self, flex: float) -> float:
        g = self.firm_globals
        return g.flex_cost_slope * flex + g.fixed_cost

    def validate(self) -> List[str]:
        errs = self.market.validate()
        if not self.firms:
            errs.append("At least one firm is required.")
        for i, f in enumerate(self.firms, 1):
            if f.rule not in RULE_TYPES:
                errs.append(f"Firm {i}: unknown rule {f.rule!r}.")
            if f.selection not in SELECTION_RULES:
                errs.append(f"Firm {i}: unknown selection rule {f.selection!r}.")
            if f.rule == "Cournot" and not (0.0 <= f.flex <= 1.0):
                errs.append(f"Firm {i}: a Cournot firm needs flexibility between 0 and 1.")
            if f.flex < 0:
                errs.append(f"Firm {i}: flexibility cannot be negative.")
        if self.adaptive.feedback == "lookahead":
            errs.append("Adaptive feedback 'lookahead' was removed: it released H-period feedback before the periods "
                        "it covers had occurred. Use 'oracle' (the same counterfactual, released at maturity).")
        elif self.adaptive.feedback not in ADAPTIVE_FEEDBACK:
            errs.append(f"Unknown Adaptive feedback {self.adaptive.feedback!r}.")
        if int(self.adaptive.window) < 1:
            errs.append("The Adaptive judgement window must be at least one period.")
        if self.burn_in >= self.periods - 2:
            errs.append("Burn-in must end before the simulation does.")
        return errs

    def copy(self, **changes) -> "Scenario":
        s = replace(
            self,
            market=replace(self.market),
            firms=[replace(f) for f in self.firms],
            firm_globals=replace(self.firm_globals),
            adaptive=replace(self.adaptive),
            evolution=replace(self.evolution),
            structural=replace(self.structural),
        )
        return replace(s, **changes) if changes else s

    def to_dict(self) -> dict:
        return asdict(self)


def linear_flex_firms(n: int = 4, slope: float = 0.25, intercept: float = 0.0,
                      rule: str = "Bertrand", selection: str = "Always",
                      threshold: float = 25.0, desired_margin: float = 5.0,
                      foresight: float = 0.0, noise: float = 0.0) -> List[FirmSpec]:
    """Firms with flexibility phi_i = slope*i + intercept (i = 1..n, most rigid first)."""
    return [FirmSpec(rule=rule, flex=slope * i + intercept, selection=selection,
                     threshold=threshold, desired_margin=desired_margin,
                     foresight=foresight, noise=noise) for i in range(1, n + 1)]


def default_scenario(**kw) -> Scenario:
    return Scenario(firms=linear_flex_firms(), **kw)
