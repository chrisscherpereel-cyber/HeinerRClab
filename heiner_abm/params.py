"""Parameter containers for the Heiner cobweb agent-based model.

Defaults reproduce the 2006 VBA workbook (HeinerIOExp.xls, SetUpForm defaults):
    MaxPrice 100, MinPrice 10, QuantityRange 1500, InitRawMatCost 45,
    MaxRawMatCost 80, MaxRawMatCostChange 10, InitProduction 200,
    MinProduction 15, FlexSlope 0.25, FlexIntercept 0, DesiredProfit 5,
    Threshold 25, four firms, record after period 25.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import List, Optional

RULE_TYPES = ("Bertrand", "Cournot")
SELECTION_RULES = ("Always", "Never", "Small", "Large", "Adaptive")

SELECTION_HELP = {
    "Always": "Always adopt the Cournot/Bertrand recommendation (maximally flexible).",
    "Never": "Never deviate from the default rule B: keep last period's production (fully rigid).",
    "Small": "SR1 - deviate only when the recommended change |q* - q| < threshold (change is risky).",
    "Large": "SR2 - deviate only when |q* - q| > threshold (large signals are clear signals).",
    "Adaptive": "Learns, per size-of-change bin, whether deviating has paid off (ex-post counterfactual "
                "payoffs) and deviates only where its learned net gain is positive.",
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
    rule: str = "Bertrand"        # production rule that generates the recommendation q*
    flex: float = 0.25            # phi: Cournot weight (0..1) or Bertrand margin sensitivity
    selection: str = "Always"     # selection rule for accepting q* vs default rule B (hold q)
    threshold: float = 25.0       # threshold for Small/Large selection rules
    desired_margin: float = 5.0   # m*: Bertrand desired profit margin (competition intensity)
    foresight: float = 0.0        # kappa: competence - fraction of next cost change anticipated
    noise: float = 0.0            # sigma: perception noise on the cost estimate
    label: Optional[str] = None


@dataclass
class GlobalFirmParams:
    q0: float = 200.0             # initial production of every firm
    q_min: float = 15.0           # minimum production
    flex_cost_slope: float = 0.0  # a: per-period flexibility cost = a*phi + b
    fixed_cost: float = 0.0       # b: per-period fixed cost (VBA FlexCostIntercept)
    margin_includes_fixed: bool = False  # Bertrand margin uses P - c - F/q (paper) vs P - c (VBA)


@dataclass
class AdaptiveParams:
    bin_edges: tuple = (5.0, 15.0, 30.0, 60.0)   # |q* - q| bin boundaries
    memory: float = 0.97                          # exponential forgetting (lambda)


@dataclass
class EvolutionParams:
    enabled: bool = False
    every: int = 50               # periods between flexibility revisions
    imitation_prob: float = 0.5   # prob. a firm copies the most profitable firm's phi
    mutation_sd: float = 0.03     # experimentation noise on phi
    flex_min: float = 0.0
    flex_max: float = 1.0


@dataclass
class Scenario:
    market: MarketParams = field(default_factory=MarketParams)
    firms: List[FirmSpec] = field(default_factory=list)
    firm_globals: GlobalFirmParams = field(default_factory=GlobalFirmParams)
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)
    evolution: EvolutionParams = field(default_factory=EvolutionParams)
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
        )
        return replace(s, **changes) if changes else s

    def to_dict(self) -> dict:
        return asdict(self)


def linear_flex_firms(n: int = 4, slope: float = 0.25, intercept: float = 0.0,
                      rule: str = "Bertrand", selection: str = "Always",
                      threshold: float = 25.0, desired_margin: float = 5.0,
                      foresight: float = 0.0, noise: float = 0.0) -> List[FirmSpec]:
    """Firms with flexibility phi_i = slope*i + intercept (the VBA convention, i = 1..n)."""
    return [FirmSpec(rule=rule, flex=slope * i + intercept, selection=selection,
                     threshold=threshold, desired_margin=desired_margin,
                     foresight=foresight, noise=noise) for i in range(1, n + 1)]


def default_scenario(**kw) -> Scenario:
    return Scenario(firms=linear_flex_firms(), **kw)
