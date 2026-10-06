"""Information and feedback: one specification shared by every simulation engine.

InfoSpec says what agents may observe and what feedback they receive. Each engine declares which specifications it can
run (ENGINES); each agent declares what it needs (AGENTS). compatibility() checks an agent against a specification and
an engine and never grants information that the specification withholds: an agent that needs unavailable information
is reported as unsupported, with the reason, so that the comparison is disabled rather than run with extra
information.

Feedback treatments
    chosen     the agent observes only the payoff of the action it took.
    full       the payoffs of alternative actions are observable, but only where the experimental design makes them
               available (for example a two-option task that shows both outcomes, or uncensored demand in an inventory
               task). The cobweb market publishes no payoffs of untaken actions, so it does not offer this treatment.
    estimated  the agent estimates the payoffs of alternatives from its own fitted (believed) model and its
               observations (in the market: rivals' observed output, realized costs and its believed demand curve).
    oracle     ORACLE, a diagnostic benchmark: true counterfactual values computed by the researcher (forked markets with
               the true demand curve and rivals' simulated reactions, or true expected gains from independent runs). No
               agent could compute them; results with it must be labeled as oracle results.

Information controls
    obs_noise              s.d. of noise on every observed market price and market quantity (separate random stream).
    obs_delay              periods before a period's market outcomes (price, market quantity, cost, payoffs, rivals) are
                           observed; feedback is released that many periods after it matures.
    demand_knowledge       "believed": the baseline demand curve, updated L periods after a shift (default);
                           "true": the true current curve (removes misspecification by fiat; ORACLE information);
                           "none": no demand model (rules that need one are unsupported).
    regime_announced       shifts are announced when they take effect, with the new demand curve.
    rivals_visible         rivals' individual outputs and payoffs are observable (aggregate output always is).
    unmet_demand_observed  censored outcomes such as unmet demand are observable (relevant where demand can exceed
                           supply, e.g. an inventory task; the cobweb market clears every period).

The Observation object is what an agent receives when it decides. It is built only from periods that have been
observed under the specification and contains no researcher-only quantity (no true demand curve unless the
specification grants it, no counterfactual, no future cost or shock).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

class UnsupportedInformation(ValueError):
    """An agent needs information or feedback that the specification (or the engine) does not provide. The run is not
    performed; callers disable the comparison and say why."""


FEEDBACK = ("chosen", "full", "estimated", "oracle")
FEEDBACK_LABELS = {"chosen": "Chosen-action feedback", "full": "Full feedback (where the design provides it)",
                   "estimated": "Estimated counterfactual", "oracle": "ORACLE counterfactual (diagnostic benchmark)"}
# What each treatment makes available. Every agent always observes the payoff of its own action; an agent may always
# form its own estimates from what it observes; full and oracle feedback add true payoffs of alternatives.
AVAILABLE: Dict[str, Tuple[str, ...]] = {"chosen": ("chosen",), "estimated": ("chosen", "estimated"),
                                         "full": ("chosen", "estimated", "full"),
                                         "oracle": ("chosen", "estimated", "oracle")}
DEMAND_KNOWLEDGE = ("none", "believed", "true")
DEMAND_LABELS = {"none": "No demand model", "believed": "Believed curve (updated L periods after a shift)",
                 "true": "True current curve (ORACLE information)"}


@dataclass
class InfoSpec:
    feedback: str = "estimated"
    obs_noise: float = 0.0
    obs_delay: int = 0
    demand_knowledge: str = "believed"
    regime_announced: bool = False
    rivals_visible: bool = True
    unmet_demand_observed: bool = True

    def validate(self) -> List[str]:
        errs = []
        if self.feedback == "lookahead":
            errs.append("Feedback 'lookahead' was removed: it released H-period feedback before the periods it covers "
                        "had occurred. Use 'oracle' (the same counterfactual, released at maturity).")
        elif self.feedback not in FEEDBACK:
            errs.append(f"Unknown feedback treatment {self.feedback!r} (choose one of {', '.join(FEEDBACK)}).")
        if self.demand_knowledge not in DEMAND_KNOWLEDGE:
            errs.append(f"Unknown demand knowledge {self.demand_knowledge!r}.")
        if self.obs_noise < 0:
            errs.append("Observation noise cannot be negative.")
        if int(self.obs_delay) < 0:
            errs.append("Observation delay cannot be negative.")
        return errs

    @property
    def is_oracle(self) -> bool:
        return self.feedback == "oracle" or self.demand_knowledge == "true"

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ observation
@dataclass(frozen=True)
class Observation:
    """Everything an agent may use when it decides in `period`. Built by an engine from observed history only."""
    period: int
    observed_period: int                 # latest period whose market outcomes are observed (period - 1 - delay)
    own_output: float                    # its current output (its own action is always known)
    own_output_then: float               # its output in observed_period
    own_payoff: float                    # its payoff in observed_period
    price: float                         # observed price of observed_period (with observation noise)
    market_quantity: float               # observed market output of observed_period (with observation noise)
    observed_cost: float                 # raw-material cost of observed_period
    cost_estimate: float                 # its estimate of this period's cost: observed cost + κ·(coming change) + σ·ε
    demand_model: Optional[Tuple[float, float]]                 # (p_max, slope) it may use, or None
    regime_announced: Optional[bool]     # under announcement: whether a shift took effect this period; else None
    rivals: Optional[Tuple[Tuple[float, float], ...]]           # (output, payoff) of each rival, if visible

    @property
    def rivals_output(self) -> float:
        """Rivals' total output in observed_period (always available from the market total)."""
        return self.market_quantity - self.own_output_then


OBSERVATION_FIELDS = tuple(Observation.__dataclass_fields__)
# Researcher-only quantities that an Observation must never carry.
RESEARCHER_ONLY = ("true_demand", "counterfactual", "profit_rule", "profit_default", "future_cost", "shift_schedule",
                   "learned_gain_of_rivals", "oracle_gain")


# ------------------------------------------------------------------------------------------------ engines
@dataclass(frozen=True)
class EngineSupport:
    key: str
    name: str
    feedback: Tuple[str, ...]            # treatments the engine can deliver
    controls: Dict[str, str]             # control -> "supported" or an explanation of the limit
    native: InfoSpec                     # the specification its registered results use
    fixed: bool = False                  # True: only the native specification can be run

    def check(self, spec: InfoSpec) -> List[str]:
        """Why this engine cannot run `spec` (empty if it can)."""
        errs = list(spec.validate())
        if spec.feedback not in self.feedback:
            errs.append(f"{self.name} cannot deliver {FEEDBACK_LABELS.get(spec.feedback, spec.feedback).lower()}: "
                        + self.controls.get("feedback", "not available in this design."))
        if self.fixed:
            for k, v in asdict(spec).items():
                if k != "feedback" and v != getattr(self.native, k):
                    errs.append(f"{self.name} runs only its native specification; '{k}' = {v!r} is not implemented "
                                f"(native {getattr(self.native, k)!r}).")
        elif not spec.unmet_demand_observed and self.controls.get("unmet_demand_observed", "").startswith("n/a"):
            pass   # not applicable: nothing is censored in this engine
        return errs


ENGINES: Dict[str, EngineSupport] = {
    "market": EngineSupport(
        "market", "Cobweb market (Market lab, hypothesis tests, competing theories)", ("chosen", "estimated", "oracle"),
        {"feedback": "the market publishes no payoffs of untaken actions, so full feedback is not available.",
         "obs_noise": "supported", "obs_delay": "supported", "demand_knowledge": "supported",
         "regime_announced": "supported", "rivals_visible": "supported (needed by evolution of flexibility)",
         "unmet_demand_observed": "n/a: the market clears every period, so nothing is censored"},
        InfoSpec()),
    "arena": EngineSupport(
        "arena", "Agent tournament engine (tournament, mechanisms, rule choice, field patterns, signature tests, "
                 "Play the market)", ("chosen", "estimated", "oracle"),
        {"feedback": "the market publishes no payoffs of untaken actions, so full feedback is not available.",
         "obs_noise": "native only (σ is perception noise on cost)", "obs_delay": "native only (no delay)",
         "demand_knowledge": "native only (believed curve)", "regime_announced": "native only (not announced)",
         "rivals_visible": "native only (visible; imitators use it)",
         "unmet_demand_observed": "n/a: the market clears every period"},
        InfoSpec(feedback="estimated"), fixed=True),
    "tasks": EngineSupport(
        "tasks", "Generalization tasks (inventory, learning, investment)", ("full", "oracle"),
        {"feedback": "the tasks show the outcome of both actions (full feedback by design).",
         "obs_noise": "native only (observation error is a task parameter)", "obs_delay": "native only (no delay)",
         "demand_knowledge": "native only", "regime_announced": "native only (shifts not announced)",
         "rivals_visible": "n/a: single decision maker",
         "unmet_demand_observed": "native only (demand observed uncensored)"},
        InfoSpec(feedback="full"), fixed=True),
    "tracking": EngineSupport(
        "tracking", "Solvable tracking benchmark", ("chosen",),
        {"feedback": "the rules are tuned in advance and do not learn; they see their own loss only.",
         "obs_noise": "native only (the observation noise r is the benchmark's parameter)",
         "obs_delay": "native only (no delay)", "demand_knowledge": "n/a", "regime_announced": "n/a",
         "rivals_visible": "n/a: single decision maker", "unmet_demand_observed": "n/a"},
        InfoSpec(feedback="chosen"), fixed=True),
}


# ------------------------------------------------------------------------------------------------ agents
@dataclass(frozen=True)
class Needs:
    key: str
    name: str
    engine: str
    feedback: Tuple[str, ...] = ()       # kinds of feedback it can learn from (any one suffices); () = no learning
    demand_model: bool = False           # needs a demand curve (believed or true) to decide
    estimate_needs_demand: bool = False  # needs a demand curve to estimate counterfactuals (estimated feedback)
    rivals_visible: bool = False         # needs rivals' individual outputs or payoffs
    uses: str = ""                       # what it actually reads
    oracle: bool = False                 # an oracle / researcher variant by design
    alternative: Dict[str, str] = field(default_factory=dict)    # treatment -> explicitly named alternative agent

    def access(self, spec: InfoSpec) -> Dict[str, str]:
        """What this agent receives under `spec`."""
        d = spec.obs_delay
        lag = f" (delayed {d})" if d else ""
        noisy = f", noise s.d. {spec.obs_noise:g}" if spec.obs_noise else ""
        usable = [f for f in self.feedback if f in AVAILABLE[spec.feedback]]
        best = next((f for f in ("oracle", "full", "estimated", "chosen") if f in usable), None)
        fb = "does not learn" if not self.feedback else FEEDBACK_LABELS[best] if best else "UNAVAILABLE"
        return {
            "Own action": "yes",
            "Own payoff": "yes" + lag,
            "Market price and output": ("yes" + lag + noisy),
            "Rivals' individual actions and payoffs": ("yes" + lag) if spec.rivals_visible else "no",
            "Demand model": {"none": "no", "believed": "believed (lags shifts)", "true": "true curve (ORACLE)"}[
                spec.demand_knowledge] + (", shifts announced" if spec.regime_announced else ""),
            "Unmet demand": "observed" if spec.unmet_demand_observed else "censored",
            "Feedback": fb,
        }


def _a(key, name, engine, **kw) -> Needs:
    return Needs(key, name, engine, **kw)


AGENTS: List[Needs] = [
    # market lab: production rules, selection rules, evolution
    _a("Bertrand", "Margin-feedback quantity rule", "market", uses="observed price, cost estimate, own output"),
    _a("Cournot", "Cournot best-reply quantity rule", "market", demand_model=True,
       uses="demand model, observed rivals' output, cost estimate"),
    _a("Always", "Selection: Always", "market", uses="the recommendation"),
    _a("Never", "Selection: Never (rule B)", "market", uses="nothing"),
    _a("Small", "Selection: Small (SR1)", "market", uses="size of the recommended change"),
    _a("Large", "Selection: Large (SR2)", "market", uses="size of the recommended change"),
    _a("Adaptive", "Selection: Adaptive (learned reliability) · estimated-gain gate", "market",
       feedback=("estimated", "oracle"), estimate_needs_demand=True,
       uses="learned gains per size bin; feedback about past deviations"),
    _a("Adaptive:lcb", "Selection: Adaptive · confidence-sensitive gate (proposed extension)", "market",
       feedback=("estimated", "oracle"), estimate_needs_demand=True,
       uses="mean, standard error and effective number of feedback items per size bin; adjustment cost"),
    _a("Adaptive:explore", "Selection: Adaptive · exploration-enabled gate (proposed extension)", "market",
       feedback=("chosen",),
       uses="its own realized payoffs after randomized trials (adopt or keep) per size bin; adjustment cost"),
    _a("Adaptive:oracle_table", "Selection: Adaptive · ORACLE benchmark gate", "market", feedback=("oracle",),
       oracle=True, uses="true mean advantage per size bin, estimated by the researcher from independent runs"),
    _a("evolution", "Evolution of flexibility (imitation)", "market", rivals_visible=True,
       uses="every firm's realized profit and flexibility"),
    # tournament engine
    _a("heiner_m", "Reliability condition · model-based target", "arena", feedback=("estimated",), demand_model=True,
       estimate_needs_demand=True,
       uses="believed demand, filtered cost and rivals' output; self-judged past deviations"),
    _a("heiner_p", "Reliability condition · price-based target", "arena", feedback=("estimated",), demand_model=True,
       estimate_needs_demand=True,
       uses="adaptive price expectation, believed slope; self-judged past deviations"),
    _a("heiner_m_true", "RC · model-based · judged with the true model", "arena", feedback=("oracle",),
       demand_model=True, oracle=True, uses="judges past deviations with the TRUE demand curve"),
    _a("heiner_p_true", "RC · price-based · judged with the true model", "arena", feedback=("oracle",),
       demand_model=True, oracle=True, uses="judges past deviations with the TRUE demand curve"),
    _a("heiner_m_oracle", "RC · model-based · oracle", "arena", feedback=("oracle",), demand_model=True, oracle=True,
       uses="true expected gains per size bin"),
    _a("heiner_p_oracle", "RC · price-based · oracle", "arena", feedback=("oracle",), demand_model=True, oracle=True,
       uses="true expected gains per size bin"),
    _a("opt_br", "Filtered best reply", "arena", demand_model=True, uses="believed demand, filtered forecasts"),
    _a("opt_nash", "Rational expectations (Cournot–Nash)", "arena", demand_model=True,
       uses="believed demand, number of firms, filtered cost"),
    _a("options_m", "Inaction-band heuristic · model-based target", "arena", demand_model=True,
       uses="believed demand, volatility of its own target"),
    _a("options_p", "Inaction-band heuristic · price-based target", "arena", demand_model=True,
       uses="adaptive price expectation, believed slope, volatility of its target"),
    _a("cobweb_p", "Adaptive price expectations", "arena", demand_model=True,
       uses="adaptive price expectation, believed slope"),
    _a("cobweb_q", "Adaptive quantity adjustment", "arena", demand_model=True, uses="believed demand, rivals' output"),
    _a("heur_wsls", "Win-stay, lose-shift", "arena", feedback=("chosen",), uses="own payoff changes"),
    _a("heur_markup", "Target-margin rule", "arena", uses="observed price, cost estimate"),
    _a("satis_m", "Aspiration search · model-based target", "arena", feedback=("chosen",), demand_model=True,
       uses="own payoff vs aspiration; believed demand"),
    _a("satis_p", "Aspiration search · price-based target", "arena", feedback=("chosen",), demand_model=True,
       uses="own payoff vs aspiration; adaptive price expectation"),
    _a("rl_softmax", "Softmax value learner", "arena", feedback=("chosen",), uses="own payoff after each move"),
    _a("rl_erevroth", "Erev–Roth propensity learner", "arena", feedback=("chosen",), uses="own payoff after each move"),
    _a("imit_best", "Imitate the best", "arena", rivals_visible=True, uses="rivals' outputs and payoffs"),
    _a("imit_avg", "Imitate the average", "arena", rivals_visible=True, uses="rivals' outputs"),
    _a("ruleb", "Rule B (rigid)", "arena", uses="nothing"),
    _a("human", "Human participant (Play the market)", "arena", feedback=("chosen",),
       uses="the screen: last price, market output, own output and payoff, cost estimate; no demand model and no "
            "individual rivals"),
    # generalization tasks
    _a("always", "Task: always deviate", "tasks", uses="observed outcomes"),
    _a("band", "Task: inaction band", "tasks", uses="observed outcomes, signal strength"),
    _a("rc_learned", "Task: RC, learned", "tasks", feedback=("full",),
       uses="observed payoff of both actions each period"),
    _a("rc_oracle", "Task: RC, oracle", "tasks", feedback=("oracle",), oracle=True,
       uses="true mean gain per bin from independent runs"),
    _a("task_ruleb", "Task: rule B (default)", "tasks", uses="observed outcomes"),
    # tracking benchmark
    _a("tracking", "Tracking rules (filter and restricted variants)", "tracking", uses="current noisy observation"),
]
AGENT_BY_KEY: Dict[Tuple[str, str], Needs] = {(a.engine, a.key): a for a in AGENTS}


@dataclass(frozen=True)
class Compatibility:
    status: str          # "supported", "oracle variant" (runs under its own, labeled ORACLE treatment) or "unsupported"
    reasons: Tuple[str, ...]
    alternative: Optional[str] = None


def compatibility(agent: Needs, spec: InfoSpec, engine: Optional[str] = None) -> Compatibility:
    """Whether `agent` can run under `spec` (in its engine) without receiving information the spec withholds.

    Oracle and true-model variants are explicitly named researcher designs: they always run under their own ORACLE
    treatment (never silently), so they are checked against the specification with oracle feedback and reported as
    'oracle variant'."""
    eng = ENGINES[engine or agent.engine]
    if spec.validate():
        return Compatibility("unsupported", tuple(spec.validate()))
    if agent.oracle:
        inner = compatibility(Needs(agent.key, agent.name, agent.engine, feedback=agent.feedback,
                                    demand_model=agent.demand_model, rivals_visible=agent.rivals_visible),
                              InfoSpec(**{**spec.to_dict(), "feedback": "oracle"}), engine)
        return Compatibility("oracle variant" if inner.status == "supported" else "unsupported", inner.reasons)
    reasons = list(eng.check(spec))
    usable = [f for f in agent.feedback if f in AVAILABLE.get(spec.feedback, ())]
    if agent.feedback and not usable:
        reasons.append(f"{agent.name} learns from {' or '.join(FEEDBACK_LABELS[f].lower() for f in agent.feedback)}, "
                       f"which the {FEEDBACK_LABELS.get(spec.feedback, spec.feedback).lower()} treatment does not provide.")
    if (agent.estimate_needs_demand and spec.demand_knowledge == "none" and "estimated" in usable
            and not any(f in usable for f in ("full", "oracle"))):
        reasons.append(f"{agent.name} estimates counterfactual payoffs with a demand model, and the specification "
                       "provides none.")
    if agent.demand_model and spec.demand_knowledge == "none":
        reasons.append(f"{agent.name} needs a demand model, and the specification provides none.")
    if agent.rivals_visible and not spec.rivals_visible:
        reasons.append(f"{agent.name} needs rivals' individual outputs or payoffs, which are not visible.")
    alt = agent.alternative.get(spec.feedback) if reasons else None
    return Compatibility("unsupported" if reasons else "supported", tuple(reasons), alt)


def adaptive_key(gate: str) -> str:
    """Registry key of the Adaptive rule with a given selection gate (heiner_abm.gates)."""
    return "Adaptive" if gate == "gain" else f"Adaptive:{gate}"


def market_agents(scn) -> List[Needs]:
    """The market-lab agents a scenario uses: production rules, selection rules and evolution."""
    keys = []
    for f in scn.firms:
        keys += [f.rule, adaptive_key(scn.adaptive.gate) if f.selection == "Adaptive" else f.selection]
    if scn.evolution.enabled:
        keys.append("evolution")
    return [AGENT_BY_KEY[("market", k)] for k in dict.fromkeys(keys)]


def scenario_errors(scn) -> List[str]:
    """Every reason a market scenario cannot run under its own information specification."""
    errs = []
    for agent in market_agents(scn):
        errs += list(compatibility(agent, scn.info, "market").reasons)
    return list(dict.fromkeys(errs))


def access_table(agents: Sequence[Needs], spec: InfoSpec, engine: Optional[str] = None):
    import pandas as pd
    rows = []
    for a in agents:
        c = compatibility(a, spec, engine)
        rows.append({"Agent": a.name, "Engine": ENGINES[engine or a.engine].name.split(" (")[0],
                     "Status": {"supported": "✅ supported", "oracle variant": "🔬 ORACLE variant (labeled)",
                                "unsupported": "⛔ unsupported"}[c.status],
                     **a.access(spec), "Uses": a.uses, "Why unsupported": " ".join(c.reasons),
                     "Oracle/researcher variant": "yes" if a.oracle else ""})
    return pd.DataFrame(rows)


def export_record(engine: str, spec: Optional[InfoSpec] = None, agents: Optional[Sequence[Needs]] = None) -> Dict:
    """The information-and-feedback specification of a run, for export alongside its results."""
    eng = ENGINES[engine]
    spec = spec or eng.native
    agents = list(agents) if agents is not None else [a for a in AGENTS if a.engine == engine]
    return {"engine": eng.name, "engine_key": engine, "spec": spec.to_dict(),
            "spec_labels": {"feedback": FEEDBACK_LABELS[spec.feedback],
                            "demand_knowledge": DEMAND_LABELS[spec.demand_knowledge]},
            "oracle_information": spec.is_oracle or any(a.oracle for a in agents),
            "engine_controls": dict(eng.controls), "engine_runs_native_only": eng.fixed,
            "agents": [{"key": a.key, "name": a.name, "access": a.access(spec), "uses": a.uses, "oracle": a.oracle,
                        "status": compatibility(a, spec, engine).status,
                        "reasons": list(compatibility(a, spec, engine).reasons)} for a in agents]}
