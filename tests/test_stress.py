"""The stress suite: that the processes are what they claim, and that the measurements mean what they say.

Three things here are worth more than the rest. That no agent can see the outcome it is about to be judged on.
That the oracle is the best FEASIBLE order rather than hindsight, which is checked by simulating the law it
claims to invert. And that recovery, which is a survival time, is estimated like one.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm import stress as S

LONG = 4000


# ============================================================================== the design is what it claims
def test_every_family_is_documented():
    for key, p in S.PROCESSES.items():
        assert p.key == key and p.name.strip() and p.variants
        assert len(p.equations.strip()) > 40, f"{key} has no equations"
        assert len(p.breaks.strip()) > 20, f"{key} does not say what it breaks"
        assert p.information.strip(), f"{key} does not say what the agent sees"
        assert p.kind in ("control", "stress")


def test_a_reserved_variant_is_never_tuned_on():
    """The brief's requirement: not every scenario may be used for tuning."""
    for key, p in S.PROCESSES.items():
        tuning, reserved = set(S.TUNING_VARIANTS[key]), set(S.EVALUATION_VARIANTS[key])
        assert tuning and reserved
        assert tuning <= set(p.variants) and reserved <= set(p.variants)
        if p.kind == "stress":
            assert not (tuning & reserved), f"{key} tunes on its reserved variant"


def test_the_control_is_marked_as_one():
    assert S.PROCESSES["base"].kind == "control"
    assert sum(p.kind == "stress" for p in S.PROCESSES.values()) == 5


def test_the_specification_map_is_complete():
    for family in S.PROCESSES:
        for policy in S.POLICIES:
            status, why = S.SPECIFICATION[family][policy]
            assert status in ("correct", "estimated", "agnostic", "misspecified")
            assert why.strip(), f"{family}/{policy} gives no reason"


def test_exactly_one_model_is_correctly_specified_and_only_on_the_control():
    correct = [(f, p) for f, per in S.SPECIFICATION.items() for p, (st, _) in per.items() if st == "correct"]
    assert correct == [("base", "bayes_true")]
    for policy in S.POLICIES:
        assert S.SPECIFICATION["decision_dependent"][policy][0] == "misspecified"


def test_the_factorial_is_a_factorial():
    cells = S.factorial("base", "standard", 200, 20)
    assert len(cells) == 32 and len({lab for lab, _ in cells}) == 32
    for factor in S.FACTOR_NAMES:
        pairs = S.matched_pairs(cells, factor)
        assert len(pairs) == 16
        j = S.FACTOR_NAMES.index(factor)
        for lo, hi in pairs:
            assert lo[j] == "-" and hi[j] == "+"
            assert lo[:j] + lo[j + 1:] == hi[:j] + hi[j + 1:], "a matched pair must differ in one factor only"


@pytest.mark.parametrize("bad", [dict(family="nope"), dict(variant="nope"), dict(sigma=-1.0),
                                 dict(hazard=1.5), dict(burn_in=999)])
def test_an_impossible_cell_is_refused(bad):
    kw = dict(family="base", variant="standard", periods=200, burn_in=20)
    kw.update(bad)
    with pytest.raises(ValueError):
        S.StressEnv(**kw)


# ============================================================================== information
class _Recorder(S.Policy):
    """Records how much it had been told each time it was asked to order."""
    key = "recorder"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.told = []
        self.known_when_asked = []

    def order(self):
        self.known_when_asked.append(len(self.told))
        return 100.0

    def update(self, y):
        self.told.append(float(y))


@pytest.mark.parametrize("family,variant", [("base", "standard"), ("heavy", "contaminated"),
                                            ("multimodal", "two_mode"), ("outside_support", "upper"),
                                            ("decision_dependent", "level"),
                                            ("decision_dependent", "regime")])
def test_no_agent_can_see_the_outcome_it_is_judged_on(family, variant):
    """Directly: when the engine asks for the order of period t, the agent has been told exactly t observations
    - y_0 to y_{t-1} and nothing else. Checked for the decision-dependent families too, where the order and the
    outcome are produced in the same loop and the mistake would be easiest to make."""
    env = S.StressEnv(family, variant, periods=150, burn_in=20)
    made = []

    class Probe(_Recorder):
        def __init__(self, e, c):
            super().__init__(e, c)
            made.append(self)

    S.POLICIES["recorder"] = Probe
    try:
        frame = S.simulate(env, "recorder", {}, 1)["path"]
    finally:
        del S.POLICIES["recorder"]

    assert len(made) == 1
    agent = made[0]
    assert np.allclose(frame["order"], 100.0), "the recording policy was not the one that acted"
    assert agent.known_when_asked == list(range(env.periods)),         "the agent had seen a different number of observations than the periods already past"
    assert np.allclose(agent.told, frame["obs"].to_numpy()), "the agent was told something other than y_t"


def test_a_policy_is_given_no_true_parameter_it_has_not_been_tuned_for():
    """bayes_est must use its configuration, not the environment. Changing the truth while holding the
    configuration must leave its ORDERS alone (the payoff changes, of course)."""
    cfg = {"hazard": 0.01, "sigma": 18.0, "tau": 8.0}
    e1 = S.StressEnv("base", "standard", periods=150, burn_in=20, sigma=12.0)
    p1 = S.simulate(e1, "bayes_est", cfg, 5)["path"]
    p2 = S.simulate(S.replace(e1, hazard=0.2), "bayes_est", cfg, 5)["path"]
    # the demand path differs, so compare the first period, which precedes any observation
    assert p1["order"].iloc[0] == pytest.approx(p2["order"].iloc[0])


def test_bayes_true_does_use_the_true_parameters():
    e1 = S.StressEnv("base", "standard", periods=150, burn_in=20, sigma=10.0)
    e2 = S.replace(e1, sigma=28.0)
    assert S.simulate(e2, "bayes_true", {}, 5)["path"]["order"].iloc[0] > \
           S.simulate(e1, "bayes_true", {}, 5)["path"]["order"].iloc[0]


# ============================================================================== the benchmark
def test_the_oracle_knows_the_state_and_not_the_draw():
    env = S.StressEnv("base", "standard", periods=300, burn_in=20)
    p = S.simulate(env, "rule_b", {}, 2)["path"]
    same_mu = p.groupby("mu")["oracle_order"].nunique()
    assert (same_mu == 1).all(), "the oracle gave two different orders in the same state"
    assert (p["oracle_order"] < p["demand"]).mean() > 0.05, "an oracle with no shortages is hindsight"


@pytest.mark.parametrize("family,variant,kw", [
    ("base", "standard", {}),
    ("heavy", "contaminated", dict(magnitude=60.0)),
    ("heavy", "student_t", {}),
    ("multimodal", "two_mode", dict(magnitude=55.0, hazard=0.004)),
    ("outside_support", "upper", dict(hazard=0.03, magnitude=40.0)),
])
def test_the_oracle_really_is_the_eighty_percent_quantile(family, variant, kw):
    """The oracle inverts a conditional law written by hand for each family. The check is empirical: order it,
    and the realized demand should fall at or below it 80% of the time."""
    env = S.StressEnv(family, variant, periods=LONG, burn_in=200, **kw)
    p = S.simulate(env, "rule_b", {}, 11)["path"]
    # Family C has two phases with different laws, and in the history phase the law has an ATOM at the clip, so
    # the 0.8 quantile is covered with probability 1 and an empirical coverage check cannot apply there. The
    # live phase has a continuous law, and is the one the family is about.
    p = p.iloc[(int(S.HISTORY_FRACTION * env.periods) + 50) if family == "outside_support" else 200:]
    served = float((p["demand"] <= p["oracle_order"] + 1e-9).mean())
    assert abs(served - S.FRACTILE) < 0.035, f"{family}/{variant}: oracle covered {served:.3f}"


def test_the_oracle_beats_every_agent_on_average():
    env = S.StressEnv("base", "standard", periods=600, burn_in=60)
    for policy in S.POLICIES:
        cfg = {"gain": 0.1, "window": 60, "rho": 0.5, "sd_scale": 1.0,
               "hazard": 0.01, "sigma": 18.0, "tau": 8.0}
        cfg = {k: v for k, v in cfg.items() if k in S.SPACES[policy]}
        assert S.simulate(env, policy, cfg, 3)["regret"] > -1e-9, f"{policy} beat the benchmark"


# ============================================================================== the processes bite
@pytest.mark.parametrize("magnitude,sigma", [(20.0, 10.0), (20.0, 28.0), (55.0, 10.0), (55.0, 28.0)])
def test_the_heavy_family_is_heavy_in_every_corner_of_its_own_design(magnitude, sigma):
    """The trap: scaling the contamination by magnitude/sigma makes it SMALLER than ordinary noise wherever the
    shock is no larger than the noise - which is half the factorial - and the family then tests nothing there
    while still being called heavy-tailed. The scale is in units of sigma precisely so this cannot happen, and
    all four corners are checked."""
    from scipy.stats import kurtosis
    env = S.StressEnv("heavy", "contaminated", periods=30000, burn_in=50, hazard=0.0,
                      magnitude=magnitude, sigma=sigma)
    e = S._innovations(env, 4)
    assert kurtosis(e) > 20.0, f"innovations are not heavy: kurtosis {kurtosis(e):.1f}"
    assert S._heavy_scale(env) >= 5.0, "the contaminating draw is not large relative to the noise"


def test_the_heavy_tail_survives_into_the_decision():
    """Kurtosis of the innovation is not the point; what matters is the upper tail of DEMAND, because that is
    the side the shortage cost punishes. The zero floor truncates the lower tail, so the comparison is made
    where the decision lives."""
    sigma = 18.0
    quiet = S.StressEnv("base", "standard", periods=30000, burn_in=50, hazard=0.0, sigma=sigma)
    heavy = S.StressEnv("heavy", "contaminated", periods=30000, burn_in=50, hazard=0.0, sigma=sigma,
                        magnitude=55.0)
    dq = S.exogenous_path(quiet, 7)["demand"]
    dh = S.exogenous_path(heavy, 7)["demand"]
    assert np.percentile(dh, 99.9) > 2.0 * np.percentile(dq, 99.9) - S.MU0
    assert abs(np.median(dh) - np.median(dq)) < 0.1 * sigma, "the contamination moved the centre, not the tail"


def test_the_optimal_order_lies_outside_the_observed_history():
    """Family C exists for this. If the best order were inside the band the history fell in, an empirical model
    could reach it and the family would test nothing."""
    env = S.StressEnv("outside_support", "upper", periods=800, burn_in=40, sigma=18.0, hazard=0.02)
    p = S.simulate(env, "rule_b", {}, 9)["path"]
    h = int(S.HISTORY_FRACTION * env.periods)
    observed_max = float(p["demand"].iloc[:h].max())
    live_order = float(p["oracle_order"].iloc[h + 10])
    assert live_order > observed_max, (live_order, observed_max)
    assert float(p["demand"].iloc[:h].std()) < float(p["demand"].iloc[h:].std())


def test_the_multimodal_mean_is_not_the_multimodal_answer():
    """With separated modes the mixture mean is a demand that almost never occurs, so ordering it is wrong."""
    env = S.StressEnv("multimodal", "two_mode", periods=1500, burn_in=50, magnitude=60.0, sigma=8.0,
                      hazard=0.001)
    p = S.simulate(env, "rule_b", {}, 13)["path"].iloc[50:]
    near_mean = float((np.abs(p["demand"] - S.MU0) < 8.0).mean())
    assert near_mean < 0.25, f"the modes are not separated: {near_mean:.3f} of demand sits near the mean"
    assert abs(float(p["oracle_order"].mean()) - S.MU0) > 15.0


def test_three_modes_differ_from_two():
    a = S.StressEnv("multimodal", "two_mode", periods=600, burn_in=40, magnitude=60.0)
    b = S.StressEnv("multimodal", "three_mode", periods=600, burn_in=40, magnitude=60.0)
    assert not np.allclose(S.exogenous_path(a, 2)["demand"], S.exogenous_path(b, 2)["demand"])


def test_the_reversal_reverses_and_is_not_announced():
    env = S.StressEnv("reversal", "single", periods=600, burn_in=40, magnitude=50.0, hazard=0.0)
    path = S.exogenous_path(env, 6)
    mu, change = path["mu"], path["change"]
    c = int(np.flatnonzero(change)[0])
    assert S.STABLE_LO * env.periods <= c < S.STABLE_HI * env.periods
    assert abs(mu[c] - mu[c - 1]) > 30.0
    assert np.ptp(mu[:c]) == 0.0, "something moved before the reversal"


# ============================================================================== decision dependence
@pytest.mark.parametrize("family,variant", [("base", "standard"), ("reversal", "single"),
                                            ("heavy", "contaminated"), ("outside_support", "upper"),
                                            ("multimodal", "two_mode")])
def test_an_exogenous_family_is_exogenous(family, variant):
    """Whatever the agent does, the demand path is the same one. This is the assumption family E breaks, so it
    has to be true everywhere else or the contrast means nothing."""
    env = S.StressEnv(family, variant, periods=250, burn_in=30)
    a = S.simulate(env, "rule_b", {}, 1)["path"]
    b = S.simulate(env, "saa", {"window": 30}, 1)["path"]
    assert np.allclose(a["demand"], b["demand"]) and np.allclose(a["mu"], b["mu"])


@pytest.mark.parametrize("variant", ["level", "regime"])
def test_the_decision_dependent_family_is_not(variant):
    env = S.StressEnv("decision_dependent", variant, periods=400, burn_in=40, magnitude=40.0)
    a = S.simulate(env, "rule_b", {}, 1)["path"]
    b = S.simulate(env, "bayes_est", {"hazard": 0.01, "sigma": 18.0, "tau": 8.0}, 1)["path"]
    assert not np.allclose(a["mu"], b["mu"]), "the agent's orders left the process untouched"


def test_the_two_oracles_agree_only_when_the_path_is_exogenous():
    ex = S.StressEnv("base", "standard", periods=250, burn_in=30)
    a = S.simulate(ex, "rule_b", {}, 2)["path"]["oracle_order"].to_numpy()
    b = S.simulate(ex, "rule_b", {}, 2, free_oracle=True)["path"]["oracle_order"].to_numpy()
    assert np.allclose(a, b)

    dd = S.StressEnv("decision_dependent", "level", periods=400, burn_in=40, magnitude=40.0)
    c = S.simulate(dd, "rule_b", {}, 2)["path"]["oracle_order"].to_numpy()
    d = S.simulate(dd, "rule_b", {}, 2, free_oracle=True)["path"]["oracle_order"].to_numpy()
    assert not np.allclose(c, d), "the free oracle faced the same process it would have changed"


def test_goodwill_recovers():
    """Without recovery every agent drives demand to the lower clip and the family measures the clip. The stock
    must decay back, so a run that stocks out rarely keeps a higher level than one that stocks out often."""
    env = S.StressEnv("decision_dependent", "level", periods=600, burn_in=60, magnitude=30.0)
    greedy = S.simulate(env, "rule_b", {}, 3)["path"].iloc[60:]
    assert greedy["mu"].mean() > S.MU_LO + 20.0, "the level collapsed to the clip"


# ============================================================================== recovery
def test_recovery_windows_are_aligned_after_the_change():
    """A window straddling the change would mix the two regimes. The earliest recovery is therefore one full
    window after the change, never sooner."""
    regret = np.concatenate([np.zeros(60), np.full(60, 10.0)])
    regret[90:] = 0.0                                     # genuinely recovered at 90
    eps, _ = S.recovery_episodes(regret, [60], window=10, tol=0.5, start=0, end=120)
    assert len(eps) == 1
    duration, event = eps[0]
    assert event == 1 and duration >= 10
    assert duration == 40, duration                       # first fully-post-change window at or after 99


def test_an_interrupted_episode_is_censored_not_counted_as_slow():
    regret = np.concatenate([np.zeros(60), np.full(200, 10.0)])
    eps, unmeasurable = S.recovery_episodes(regret, [60, 100], window=10, tol=0.5, start=0, end=260)
    assert eps == [(40, 0)], eps                           # cut short by the change at 100
    assert unmeasurable == 1, "the second baseline was never a settled level"


def test_an_agent_that_never_recovers_does_not_score_fast_recoveries():
    """The trap this guards: with a contaminated baseline, permanent failure looks like instant recovery."""
    never = np.concatenate([np.zeros(40), np.full(300, 20.0)])
    eps, unmeasurable = S.recovery_episodes(never, [40, 120, 200, 280], window=10, tol=0.5, start=0, end=340)
    assert [e for _, e in eps] == [0], eps
    assert unmeasurable == 3
    km = S.kaplan_meier([d for d, _ in eps], [e for _, e in eps], unmeasurable)
    assert np.isnan(km["median_recovery"]) and km["unmeasurable"] == 3


def test_a_settled_agent_scores_every_episode():
    cycle = np.concatenate([np.zeros(60), np.full(30, 10.0), np.zeros(60), np.full(30, 10.0), np.zeros(40)])
    eps, unmeasurable = S.recovery_episodes(cycle, [60, 150], window=10, tol=0.5, start=0, end=220)
    assert eps == [(40, 1), (40, 1)] and unmeasurable == 0


def test_an_episode_with_no_room_to_measure_is_not_invented():
    regret = np.concatenate([np.zeros(60), np.full(60, 10.0)])
    eps, unmeasurable = S.recovery_episodes(regret, [60, 64], window=20, tol=0.5, start=0, end=120)
    assert eps[0][1] == 0 and unmeasurable == 1


def test_kaplan_meier_matches_a_hand_computation():
    # durations 2(event), 3(censored), 5(event), 6(event): S drops 1 -> .75 -> .375 -> 0
    km = S.kaplan_meier([2, 3, 5, 6], [1, 0, 1, 1])
    assert km["episodes"] == 4
    assert km["censoring_rate"] == pytest.approx(0.25)
    assert km["median_recovery"] == pytest.approx(5.0)


def test_a_median_that_is_not_estimable_is_not_invented():
    """Mostly-censored episodes must give NaN, not the largest duration seen, which would read as a fast
    recovery precisely when the agent never recovered."""
    km = S.kaplan_meier([5, 6, 7], [0, 0, 0])
    assert np.isnan(km["median_recovery"])
    assert km["censoring_rate"] == pytest.approx(1.0)
    assert np.isnan(S.kaplan_meier([], [])["median_recovery"])


def test_a_faster_agent_recovers_sooner():
    env = S.StressEnv("reversal", "single", periods=700, burn_in=40, magnitude=50.0, hazard=0.0, tau=2.0)
    slow = S.recovery(env, S.simulate(env, "rule_b", {}, 5)["path"], window=20, tol_fraction=0.25)
    fast = S.recovery(env, S.simulate(env, "ewma", {"gain": 0.5}, 5)["path"], window=20, tol_fraction=0.25)
    assert slow["episodes"] == fast["episodes"] == 1
    assert fast["median_recovery"] <= slow["median_recovery"] or np.isnan(slow["median_recovery"])


# ============================================================================== metrics
def test_the_metrics_are_what_they_are_called():
    env = S.StressEnv("base", "standard", periods=300, burn_in=50)
    r = S.simulate(env, "rule_b", {}, 1)
    p = r["path"].iloc[50:]
    assert r["cum_net_payoff"] == pytest.approx(p["net"].sum())
    assert r["mean_net_payoff"] == pytest.approx(p["net"].mean())
    assert r["tail_loss_cvar5"] <= r["mean_net_payoff"]
    assert r["worst_period"] <= r["tail_loss_cvar5"]
    assert r["service_level"] + r["stockout_rate"] == pytest.approx(1.0)
    assert r["calibration_gap"] == pytest.approx(r["service_level"] - S.FRACTILE)
    assert r["regret"] == pytest.approx(p["regret"].mean())


def test_an_adjustment_cost_is_charged_for_moving_and_only_for_moving():
    free = S.StressEnv("base", "standard", periods=200, burn_in=20, switch_cost=0.0)
    paid = S.replace(free, switch_cost=6.0)
    a, b = S.simulate(free, "ewma", {"gain": 0.3}, 1), S.simulate(paid, "ewma", {"gain": 0.3}, 1)
    assert a["adjustment_cost"] == 0.0 and b["adjustment_cost"] > 0.0
    assert np.allclose(a["path"]["order"], b["path"]["order"]), "the charge changed the orders"
    pb = b["path"].iloc[20:]
    moved = np.abs(np.diff(b["path"]["order"].to_numpy()))[19:] > S.ACTION_TOL
    assert np.allclose(pb["adjustment"].to_numpy()[:len(moved)] > 0, moved)


# ============================================================================== plan and budget
def test_the_plan_is_frozen_and_covers_what_it_must():
    a, b = S.StressPlan(), S.StressPlan()
    assert a.digest == b.digest and len(a.digest) == 16
    assert S.StressPlan(periods=601).digest != a.digest
    with pytest.raises(Exception):
        a.paths = 9
    blob = a.to_json()
    for needle in ("specification", "equations", "factors", "HEAVY_RATE", "reversal"):
        assert needle in blob


def test_the_plan_hash_moves_if_a_factor_level_or_the_specification_changes():
    before = S.StressPlan().digest
    S.FACTORS["sigma"] = (11.0, 28.0)
    try:
        assert S.StressPlan().digest != before
    finally:
        S.FACTORS["sigma"] = (10.0, 28.0)
    assert S.StressPlan().digest == before


def test_tuning_and_evaluation_seeds_never_meet():
    plan = S.StressPlan()
    tuning = {S._path_seed(plan, plan.seed_block_tuning, f, lab, k)
              for f in plan.families for lab, _ in S.factorial(f, "standard" if f == "base" else
                                                               S.TUNING_VARIANTS[f][0], 200, 20)
              for k in range(plan.tuning_paths)}
    evaluation = {S._path_seed(plan, plan.seed_block_evaluation, f, lab, k)
                  for f in plan.families for lab, _ in S.factorial(f, "standard" if f == "base" else
                                                                   S.EVALUATION_VARIANTS[f][0], 200, 20)
                  for k in range(plan.paths)}
    assert tuning and evaluation and not (tuning & evaluation)


def test_every_tunable_policy_gets_the_same_budget():
    t = S.budget_table(S.StressPlan())
    tunable = t[t["parameters"] > 0]
    assert tunable["configurations"].nunique() == 1
    assert tunable["runs_per_configuration"].nunique() == 1
    assert (t[t["parameters"] == 0]["configurations"] == 1).all()
    assert (t[t["parameters"] == 0]["note"] != "").all()


def test_sampled_configurations_stay_inside_their_space():
    for policy, space in S.SPACES.items():
        for cfg in S.sample_configs(policy, 24, 3):
            assert set(cfg) == set(space)
            for name, v in cfg.items():
                lo, hi, _ = space[name]
                assert lo - 1e-9 <= v <= hi + 1e-9
            if "window" in cfg:
                assert isinstance(cfg["window"], int)


def test_the_boundary_report_finds_a_parameter_pinned_to_its_edge():
    rep = S.boundary_report({"ewma": {"gain": 0.8}, "saa": {"window": 40}})
    assert bool(rep[rep["parameter"] == "gain"]["at_boundary"].iloc[0])
    assert not bool(rep[rep["parameter"] == "window"]["at_boundary"].iloc[0])


# ============================================================================== end to end
def test_the_suite_runs_and_exports_what_it_promises():
    frames = S.run_suite(S.StressPlan(**S.TINY, families=("base", "multimodal")))
    assert set(frames) >= set(S.EXPORTS)
    cells = frames["cells"]
    assert len(cells) > 0 and cells["policy"].nunique() == len(S.POLICIES)
    assert set(cells["variant"]) == {"standard", "three_mode"}, "evaluation used a tuning variant"
    assert frames["factors"]["pairs"].max() >= 1
    for frame in frames.values():
        assert isinstance(frame, pd.DataFrame)


# ============================================================================== literature
def test_every_cited_source_resolves():
    from heiner_abm.literature import REFERENCES
    cited = {src for p in S.PROCESSES.values() for src in p.sources}
    assert cited, "no family cites anything"
    for key in cited:
        ref = REFERENCES[key]
        assert ref.authors.strip() and ref.year and ref.title.strip()


def test_the_borrowed_methods_are_cited_as_connections_not_implementations():
    """The suite runs none of the algorithms it cites. The module says so, and this keeps it saying so: if an
    implementation is ever added, this test should be the thing that fails."""
    doc = " ".join(S.__doc__.split())        # the docstring is wrapped; the claim is not about its line breaks
    for needle in ("DOES NOT IMPLEMENT", "a connection, not an implementation",
                   "does not claim to implement"):
        assert needle in doc, f"the module no longer states: {needle}"
    assert not hasattr(S, "catoni_detector") and not hasattr(S, "decision_dependent_ambiguity_set")


# ============================================================================== reproducibility
def test_paths_are_identical_in_a_fresh_process():
    """Python randomizes string hashing per process. If a seed ever derives from the builtin hash of a family or
    variant name, every run draws a different path and the frozen plan reproduces nothing - silently. This runs a
    SEPARATE interpreter to make sure that cannot come back."""
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, %r);"
            "from heiner_abm import stress as S;"
            "e = S.StressEnv('heavy', 'contaminated', periods=40, burn_in=5);"
            "p = S.exogenous_path(e, 3);"
            "print(round(float(p['demand'].sum()), 9), round(float(p['obs'].sum()), 9))"
            % str(__import__("pathlib").Path(S.__file__).resolve().parents[1]))
    env = S.StressEnv("heavy", "contaminated", periods=40, burn_in=5)
    here = S.exogenous_path(env, 3)
    mine = f"{round(float(here['demand'].sum()), 9)} {round(float(here['obs'].sum()), 9)}"
    seen = set()
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert seen == {mine}, f"a fresh process drew a different path: {seen} vs {mine}"


def test_the_name_tag_is_stable():
    assert S._tag("heavy") == S._tag("heavy")
    assert S._tag("heavy") != S._tag("reversal")
    assert S._tag("heavy") == 0x22d5925e, "the tag changed: every stored path would move"


def test_changing_the_path_code_invalidates_the_plan():
    """A plan hash over parameters alone would still match results produced by different code. The seed
    derivation was changed once during development without moving the hash; this makes that impossible."""
    import heiner_abm.stress as mod
    before_code, before_plan = S.stress_code_digest(), S.StressPlan().digest
    original = mod._heavy_scale
    try:
        mod._heavy_scale = lambda env: 1.0
        assert S.stress_code_digest() != before_code
        assert S.StressPlan().digest != before_plan
    finally:
        mod._heavy_scale = original
    assert S.stress_code_digest() == before_code and S.StressPlan().digest == before_plan


def test_the_code_digest_is_stable_across_processes():
    import subprocess
    import sys
    root = str(__import__("pathlib").Path(S.__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c",
                          f"import sys; sys.path.insert(0, {root!r});"
                          "from heiner_abm.stress import stress_code_digest as d; print(d())"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == S.stress_code_digest()


def test_the_documented_plan_hash_is_the_current_one():
    """Results were once published in this repository under a plan hash they were not produced by. The note and
    the README quote a hash; it has to be the hash the code actually computes."""
    from pathlib import Path
    root = Path(S.__file__).resolve().parents[1]
    current = S.StressPlan().digest
    note = (root / "docs" / "stress_suite.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"`{current}`" in note and f"`{current}`" in readme
    header = note.split("## Results")[0]
    assert f"Frozen plan **`{current}`**" in header
    # the superseded hash may appear only where the note explains that the run was discarded
    for line in note.splitlines():
        if "41646b8bbca17af2" in line:
            assert "discarded" in line, line


def test_the_results_on_disk_came_from_this_plan():
    import json
    from pathlib import Path
    spec = Path(S.__file__).resolve().parents[1] / "docs" / "stress_results" / "spec.json"
    if not spec.exists():
        pytest.skip("results not generated in this checkout")
    assert json.loads(spec.read_text(encoding="utf-8"))["plan_digest"] == S.StressPlan().digest
