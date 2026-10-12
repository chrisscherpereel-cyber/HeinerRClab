"""Decision benchmarks (heiner_abm.bench_inventory, bench_bandit, bench_tuning): correctness on solvable and enumerated
cases, information timing, the tuning protocol, and the separation of benchmarks, oracles and references from the
rankings."""
import numpy as np
import pytest

from heiner_abm import bench_bandit as BB
from heiner_abm import bench_inventory as BI
from heiner_abm import tasks
from heiner_abm.bench_checks import CHECKS, run_checks
from heiner_abm.bench_tuning import Family, Param, sample_configs, tune
from heiner_abm.information import AGENT_BY_KEY, ENGINES, compatibility
from heiner_abm.literature import REFERENCES
from heiner_abm.model_spec import all_specs


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_correctness_check_passes(check):
    cid, what, passed, err, tol = check()
    assert passed, (cid, what, err, tol)


# ------------------------------------------------------------------------------------------------ information timing
def _y(T=160, seed=1):
    env = tasks.TaskEnv("inventory", 15.0, 0.03, 5.0, 0.1, seed)
    return env, BI._stream(env, T)[2]


@pytest.mark.parametrize("make", [
    lambda y: BI.window_orders(y, 30, BI.saa_order_window, np.full(len(y), 100.0)),
    lambda y: BI.window_orders(y, 30, lambda W: BI.dro_orders_window(W, 0.3), np.full(len(y), 100.0)),
    lambda y: BI.bocpd_orders(y, 0.02, 30.0, 20.0),
    lambda y: BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=2.0),
], ids=["saa", "dro", "bocpd", "bayes_true"])
def test_inventory_orders_use_only_past_observations(make):
    _, y = _y()
    K = 90
    alt = y.copy()
    alt[K:] += np.random.default_rng(0).normal(0, 40, len(y) - K)
    a, b = make(y), make(alt)
    np.testing.assert_array_equal(a[:K + 1], b[:K + 1])          # the order of period K uses y up to K - 1
    assert not np.array_equal(a[K + 1:], b[K + 1:])


@pytest.mark.parametrize("family,cfg", [("ucb", {"c": 0.5}), ("sw_ucb", {"window": 25, "c": 0.5}),
                                        ("d_ucb", {"one_minus_gamma": 0.05, "c": 0.5}),
                                        ("eps_greedy", {"eps": 0.1, "alpha": 0.2})])
def test_bandit_choices_use_only_past_payoffs(family, cfg, monkeypatch):
    env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    T, K = 200, 120
    state, pay, obs = BB._stream(env, T)
    base = BB.actions(env, family, cfg, T)
    alt = obs.copy()
    alt[K:] += 30.0
    # the stub must take the evaluation condition too, since _stream now carries one
    monkeypatch.setattr(BB, "_stream", lambda e, t, tail="gaussian": (state, pay, alt))
    pert = BB.actions(env, family, cfg, T)
    np.testing.assert_array_equal(base[:K + 1], pert[:K + 1])


def test_every_agent_makes_one_choice_per_period_with_matched_feedback():
    env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    for k, f in BB.FAMILIES.items():
        cfg = {p: (v.lo * v.hi) ** 0.5 if v.log else (v.lo + v.hi) / 2 for p, v in f.space.items()}
        assert BB.actions(env, k, cfg, 150).shape == (150,)
        if f.role == "competitor":
            assert f.feedback == "Chosen-action payoff"
            assert AGENT_BY_KEY[("bench_bandit", BB.REGISTRY_KEY[k])].feedback == ("chosen",)
    assert BB.FAMILIES["full_ref"].role == "reference"                  # full feedback: never ranked as a bandit
    assert AGENT_BY_KEY[("tasks", "task_full_ref")].feedback == ("full",)


# ------------------------------------------------------------------------------------------------ properties
def test_worst_case_grows_with_ambiguity_and_starts_at_the_empirical_loss():
    L = np.random.default_rng(4).uniform(0, 30, 12)
    vals = [float(BI.chi2_worst_case(L, r)) for r in (0.0, 0.01, 0.1, 0.5, 2.0, 11.0, 50.0)]
    assert vals[0] == pytest.approx(L.mean(), abs=1e-12)
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(L.max())


def test_full_ambiguity_gives_the_minimax_order():
    """Once the ball contains every point mass (rho >= N - 1) the worst case is the largest loss, so the robust order
    equalizes the overage loss at the smallest and the shortage loss at the largest observation:
    S = (h * min + p * max) / (h + p)."""
    W = np.random.default_rng(8).normal(100, 25, (1, 40))
    S = float(BI.dro_orders_window(W, 39.0)[0])
    exact = (BI.H_OVER * W.min() + BI.P_SHORT * W.max()) / (BI.H_OVER + BI.P_SHORT)
    assert S == pytest.approx(exact, abs=1e-6)


def test_grid_benchmark_is_insensitive_to_refining_the_grid():
    _, y = _y(T=300)
    a = BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=1.0)
    b = BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=0.5)
    assert np.mean(np.abs(a - b)) < 0.5


def test_correct_model_beats_its_misspecified_variant_on_average():
    """Sanity check on a few environments: the correctly specified filter earns at least as much as BOCPD at a
    reasonable configuration (not a reported result)."""
    envs = tasks._env_set("inventory", 4, 991)
    cfg = {"hazard": 0.02, "prior_sd": 40.0, "obs_sd": 25.0}
    true = np.mean([BI.run_agent(e, "bayes_true", {}, 500, 50)["payoff"] for e in envs])
    mis = np.mean([BI.run_agent(e, "bocpd", cfg, 500, 50)["payoff"] for e in envs])
    assert true >= mis


# ------------------------------------------------------------------------------------------------ tuning protocol
def test_environment_sets_are_disjoint_and_tuning_is_nested():
    plan = BI.InventoryPlan(**BI.QUICK_INVENTORY)
    sets = plan.env_sets()
    seeds = [{e.seed for e in v} for v in sets.values()]
    assert not (seeds[0] & seeds[1]) and not (seeds[1] & seeds[2]) and not (seeds[0] & seeds[2])
    fam = Family("toy", "toy", "competitor", {"x": Param(0.0, 1.0)}, "", "")
    run = lambda env, cfg: {"payoff": -(cfg["x"] - 0.3) ** 2 - 0.01 * (env % 3)}
    res = tune(fam, run, [0, 1, 2], [3, 4], [5, 6, 7], (1, 2, 4, 8), reps=2, seed=1)
    for _, d in res.curve.groupby("rep"):
        assert np.all(np.diff(d.sort_values("budget")["train"].to_numpy()) >= 0)   # more budget never trains worse
    cfgs = sample_configs(fam.space, 8, 1)
    assert cfgs[:4] == sample_configs(fam.space, 4, 1)                              # budgets are prefixes
    assert len(res.test) == 3 and abs(res.final["x"] - 0.3) < 0.3


def test_quick_runs_keep_benchmarks_and_oracles_out_of_the_rankings():
    for out in (BI.run_inventory(BI.InventoryPlan(**{**BI.QUICK_INVENTORY, "periods": 250, "burn_in": 30})),
                BB.run_bandit(BB.BanditPlan(**BB.QUICK_BANDIT))):
        rk = out["ranking"]
        assert rk.loc[rk["role"] != "competitor", "rank"].isna().all()
        ranked = rk[rk["role"] == "competitor"].sort_values("rank")
        assert list(ranked["rank"]) == sorted(ranked["rank"])
        assert {"budget", "train_runs", "validation"} <= set(out["curve"].columns)


# ------------------------------------------------------------------------------------------------ registry
def test_every_family_is_registered_documented_and_sourced():
    keys = {s.key for s in all_specs()}
    for mod, engine in ((BI, "bench_inventory"), (BB, "bench_bandit")):
        for k, f in mod.FAMILIES.items():
            rk = mod.REGISTRY_KEY[k]
            eng = "tasks" if rk == "task_full_ref" else engine
            agent = AGENT_BY_KEY[(eng, rk)]
            assert rk in keys
            assert all(s in REFERENCES for s in f.sources)
            assert f.assumptions and f.compute
            status = compatibility(agent, ENGINES[eng].native, eng).status
            assert status == ("oracle variant" if f.role in ("benchmark", "oracle") else "supported"), (k, status)
            assert agent.oracle == (f.role in ("benchmark", "oracle"))


def test_checks_table():
    df = run_checks()
    assert df["passed"].all() and len(df) == len(CHECKS)


def _env(hazard=0.01, noise=20.0, obs_noise=10.0, gain=0.1, seed=4):
    """One inventory environment for the stage-A tests."""
    return tasks.TaskEnv("inventory", noise, hazard, obs_noise, gain, seed)


# ================================================================ A. Bayesian state space, estimated parameters
class TestBayesianStateSpace:
    """Stage A: latent demand variation and measurement error kept apart, parameters estimated, gaps handled."""

    H, SIG, TAU = 0.01, 20.0, 10.0

    def test_the_estimated_model_is_a_competitor_and_the_correct_one_is_not(self):
        fam = BI.FAMILIES
        assert fam["bayes_est"].role == "competitor"
        assert fam["bayes_true"].role == "benchmark"
        assert set(fam["bayes_est"].space) == {"hazard", "sigma", "tau"}
        assert fam["bayes_true"].space == {}, "a benchmark given the true parameters has nothing to tune"

    def test_the_estimated_model_is_never_handed_a_true_parameter(self):
        """The dispatch reads its parameters from the configuration; the benchmark reads them from the environment.
        Running it with a configuration far from the truth must change its orders."""
        env = _env(hazard=self.H, noise=self.SIG, obs_noise=self.TAU)
        truth = dict(hazard=self.H, sigma=self.SIG, tau=self.TAU)
        wrong = dict(hazard=0.15, sigma=5.0, tau=50.0)
        a = BI.orders(env, "bayes_est", truth, 200)
        b = BI.orders(env, "bayes_est", wrong, 200)
        assert not np.allclose(a, b), "the configuration had no effect, so it is not the source of the parameters"

    def test_with_the_true_parameters_it_reproduces_the_correctly_specified_benchmark(self):
        env = _env(hazard=self.H, noise=self.SIG, obs_noise=self.TAU)
        est = BI.orders(env, "bayes_est", dict(hazard=self.H, sigma=self.SIG, tau=self.TAU), 200)
        true = BI.orders(env, "bayes_true", {}, 200)
        assert np.array_equal(est, true), "the same filter must give the same orders from the same parameters"

    # ------------------------------------------------------------ separating the two sources of variation
    def test_the_order_uses_latent_variation_only_and_never_measurement_error(self):
        """With nothing observed the posterior cannot depend on tau, so neither can the order. Any leak of
        measurement error into the fractile margin shows up here as a difference."""
        y = np.full(40, np.nan)
        base = BI.state_space_orders(y, self.H, self.SIG, 1.0)
        for tau in (0.0, 5.0, 25.0, 100.0):
            assert np.array_equal(BI.state_space_orders(y, self.H, self.SIG, tau), base)

    def test_moving_variation_from_demand_to_measurement_lowers_the_order(self):
        """Hold the total observation noise fixed and shift it from demand to the instrument. The agent is exposed
        to less real variation, so it should hold less stock, not the same."""
        rng = np.random.default_rng(3)
        y = 100.0 + rng.normal(0, 25.0, 300)
        total = 25.0
        margins = []
        for sigma in (24.0, 18.0, 10.0):
            tau = float(np.sqrt(max(total ** 2 - sigma ** 2, 1e-6)))
            S, means = BI.state_space_orders(y, self.H, sigma, tau, return_post=True)
            margins.append(float(np.mean(S[50:] - means[50:])))
        assert margins[0] > margins[1] > margins[2], f"margin did not fall with latent variation: {margins}"

    # ------------------------------------------------------------ missing observations
    def test_a_missing_observation_advances_the_state_without_updating_it(self):
        grid = np.arange(BI.MU_LO, BI.MU_HI + 1e-9, 1.0)
        P = BI.grid_transition(grid, self.H)
        _, means = BI.state_space_orders(np.array([105.0]), self.H, self.SIG, self.TAU, return_post=True)
        w = np.zeros(len(grid))
        w[np.argmin(np.abs(grid - BI.MU0))] = 1.0
        pred = w @ P
        post = pred * BI.obs_likelihood(105.0, grid, self.SIG, self.TAU)
        assert means[0] == pytest.approx(float(post @ grid / post.sum()))
        # now the same period unobserved: the mean must be the prediction, not the update
        _, gap = BI.state_space_orders(np.array([np.nan]), self.H, self.SIG, self.TAU, return_post=True)
        assert gap[0] == pytest.approx(float(pred @ grid))

    def test_a_gap_is_not_read_as_an_average_observation(self):
        y = np.array([100.0, 160.0, 160.0, np.nan, np.nan, 160.0])
        imputed = np.where(np.isnan(y), 100.0, y)
        a = BI.state_space_orders(y, self.H, self.SIG, self.TAU)
        b = BI.state_space_orders(imputed, self.H, self.SIG, self.TAU)
        assert not np.allclose(a, b), "filling a gap with the prior mean gave the same answer as leaving it empty"

    @pytest.mark.parametrize("pattern", ["none", "alternate", "run", "all"])
    def test_every_pattern_of_missingness_is_handled(self, pattern):
        rng = np.random.default_rng(11)
        y = 100.0 + rng.normal(0, 20.0, 60)
        if pattern == "alternate":
            y[::2] = np.nan
        elif pattern == "run":
            y[10:40] = np.nan
        elif pattern == "all":
            y[:] = np.nan
        S, means = BI.state_space_orders(y, self.H, self.SIG, self.TAU, return_post=True)
        assert np.isfinite(S).all() and np.isfinite(means).all()
        assert (S >= 0).all()

    def test_observing_more_cannot_be_worse_than_observing_nothing(self):
        """A sanity check on the filter as a whole: tracking a shifted mean from data must beat ignoring the data."""
        rng = np.random.default_rng(5)
        mu = np.where(np.arange(240) < 120, 90.0, 150.0)
        d = np.maximum(0.0, mu + rng.normal(0, self.SIG, 240))
        y = d + rng.normal(0, self.TAU, 240)
        seen = BI.state_space_orders(y, self.H, self.SIG, self.TAU)
        blind = BI.state_space_orders(np.full(240, np.nan), self.H, self.SIG, self.TAU)
        pay = lambda S: float(BI.inventory_payoff(S, d)[40:].mean())
        assert pay(seen) > pay(blind)

    def test_the_family_states_its_assumptions_and_its_cost(self):
        f = BI.FAMILIES["bayes_est"]
        assert "tuned on training environments" in f.assumptions
        assert "never told" in f.assumptions
        assert f.compute.strip() and f.sources


# ================================================================ B. Thompson sampling for a switching bandit
class TestThompsonSampling:
    """Stage B: the action space and the observable feedback are the claims; both are tested directly."""

    ENV = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    T = 400
    SW = dict(window=50, reward_sd=1.0, prior_sd=1.0)
    DISC = dict(one_minus_gamma=0.02, reward_sd=1.0, prior_sd=1.0)

    # ------------------------------------------------------------ the action space
    @pytest.mark.parametrize("family,cfg", [("ts_sw", SW), ("ts_disc", DISC)])
    def test_exactly_one_arm_is_chosen_in_every_period(self, family, cfg):
        a = BB.actions(self.ENV, family, cfg, self.T)
        assert a.shape == (self.T,)
        assert set(np.unique(a)) <= set(range(BB.ARMS))
        assert a.dtype.kind in "iu"

    # ------------------------------------------------------------ the observable feedback
    @pytest.mark.parametrize("family,cfg", [("ts_sw", SW), ("ts_disc", DISC)])
    def test_only_the_chosen_option_is_observed(self, family, cfg):
        """The decisive test of the information restriction: corrupt the payoff of the option NOT chosen in every
        period. An agent restricted to chosen-action feedback cannot notice."""
        _, _, obs = BB._stream(self.ENV, self.T)
        draws = BB.posterior_draws(self.ENV, self.T)
        run = (lambda o: BB.ts_sw_actions(o, int(cfg["window"]), cfg["reward_sd"], cfg["prior_sd"], draws)) \
            if family == "ts_sw" else \
            (lambda o: BB.ts_disc_actions(o, 1.0 - cfg["one_minus_gamma"], cfg["reward_sd"], cfg["prior_sd"], draws))
        base = run(obs.copy())
        poisoned = obs.copy().astype(float)
        for t, at in enumerate(base):
            poisoned[t, 1 - int(at)] += 1000.0          # the branch it did not take
        assert np.array_equal(run(poisoned), base), "the agent reacted to a payoff it never saw"

    def test_it_is_not_told_the_regime_or_the_switch_times(self):
        """Its only inputs are the observation array, its tuned configuration and its own draws."""
        import inspect
        for fn in (BB.ts_sw_actions, BB.ts_disc_actions):
            args = set(inspect.signature(fn).parameters)
            assert not ({"state", "env", "regime", "switch"} & args), args

    # ------------------------------------------------------------ the posterior
    def test_the_posterior_is_the_gaussian_conjugate_update(self):
        s, n = np.array([4.0, 0.0]), np.array([8.0, 0.0])
        rsd, psd = 2.0, 3.0
        draw = BB._ts_posterior_draw(s, n, rsd, psd, np.zeros(2))
        prec = 1.0 / psd ** 2 + n / rsd ** 2
        assert draw == pytest.approx((s / rsd ** 2) / prec)
        assert draw[1] == pytest.approx(0.0), "an arm with no evidence must sit at the prior mean"

    def test_uncertainty_falls_as_evidence_accumulates(self):
        rsd, psd = 1.0, 2.0
        spread = []
        for n in (0.0, 5.0, 100.0):
            lo = BB._ts_posterior_draw(np.zeros(2), np.full(2, n), rsd, psd, np.full(2, -1.0))
            hi = BB._ts_posterior_draw(np.zeros(2), np.full(2, n), rsd, psd, np.full(2, +1.0))
            spread.append(float(hi[0] - lo[0]))
        assert spread[0] > spread[1] > spread[2]

    # ------------------------------------------------------------ forgetting
    def test_a_window_longer_than_the_horizon_never_forgets(self):
        _, _, obs = BB._stream(self.ENV, 120)
        draws = BB.posterior_draws(self.ENV, 120)
        long_window = BB.ts_sw_actions(obs, 10_000, 1.0, 1.0, draws)
        no_discount = BB.ts_disc_actions(obs, 1.0, 1.0, 1.0, draws)
        assert np.array_equal(long_window, no_discount), "with no forgetting the two variants must coincide"

    @pytest.mark.parametrize("family", ["ts_sw", "ts_disc"])
    def test_forgetting_faster_tracks_a_switch_faster(self, family):
        """A sharp switch at the midpoint: the quicker-forgetting setting must follow it sooner."""
        T = 400
        obs = np.zeros((T, 2))
        obs[:T // 2, 0], obs[:T // 2, 1] = 1.0, -1.0
        obs[T // 2:, 0], obs[T // 2:, 1] = -1.0, 1.0
        draws = BB.posterior_draws(self.ENV, T)
        if family == "ts_sw":
            quick = BB.ts_sw_actions(obs, 20, 1.0, 1.0, draws)
            slow = BB.ts_sw_actions(obs, 400, 1.0, 1.0, draws)
        else:
            quick = BB.ts_disc_actions(obs, 0.9, 1.0, 1.0, draws)
            slow = BB.ts_disc_actions(obs, 1.0, 1.0, 1.0, draws)
        after = slice(T // 2, T)
        assert quick[after].mean() > slow[after].mean(), "the quicker setting did not follow the switch sooner"

    # ------------------------------------------------------------ parameters are estimated, not given
    def test_the_working_parameters_are_tuned_and_not_the_environment_noise(self):
        a = BB.actions(self.ENV, "ts_sw", dict(self.SW, reward_sd=0.2), self.T)
        b = BB.actions(self.ENV, "ts_sw", dict(self.SW, reward_sd=8.0), self.T)
        assert not np.array_equal(a, b), "the working reward s.d. had no effect, so it is not being used"
        f = BB.FAMILIES["ts_sw"]
        assert {"reward_sd", "prior_sd", "window"} == set(f.space)
        assert "not told" in f.assumptions and f.role == "competitor"

    def test_the_discounted_family_labels_its_adaptation_rather_than_borrowing_authority(self):
        f = BB.FAMILIES["ts_disc"]
        assert "adaptation" in f.assumptions and "not a result" in f.assumptions
        assert "thompson1933" in f.sources and "trovo2020" not in f.sources

    def test_the_sliding_window_family_cites_the_paper_it_implements(self):
        assert "trovo2020" in BB.FAMILIES["ts_sw"].sources
        for key in ("thompson1933", "agrawal2013", "trovo2020"):
            assert key in REFERENCES, f"{key} is cited but not in the registry"

    # ------------------------------------------------------------ streams and reproducibility
    def test_thompson_draws_come_from_their_own_stream(self):
        ts = BB.posterior_draws(self.ENV, 50)
        eps = BB.exploration_draws(self.ENV, 50)
        assert ts.shape == (BB.ARMS, 50)
        assert not np.allclose(ts[:, :min(len(ts[0]), len(eps[0]))], eps[:, :50])

    @pytest.mark.parametrize("family,cfg", [("ts_sw", SW), ("ts_disc", DISC)])
    def test_a_run_reproduces(self, family, cfg):
        assert np.array_equal(BB.actions(self.ENV, family, cfg, self.T),
                              BB.actions(self.ENV, family, cfg, self.T))

    @pytest.mark.parametrize("family,cfg", [("ts_sw", SW), ("ts_disc", DISC)])
    def test_it_scores_below_the_oracle_that_is_told_the_regime(self, family, cfg):
        """The only cross-family ordering that holds without tuning. A ranking against the other competitors is a
        question for the tuning protocol, which gives every family the same training opportunities; asserting one
        here at hand-picked settings would be the comparison that protocol exists to prevent."""
        ts = BB.run_agent(self.ENV, family, cfg, 600, 50)["payoff"]
        oracle = BB.run_agent(self.ENV, "oracle", {}, 600, 50)["payoff"]
        assert np.isfinite(ts) and ts < oracle

    @pytest.mark.parametrize("family", ["ts_sw", "ts_disc"])
    def test_forgetting_pays_on_these_switching_environments(self, family):
        """A within-family comparison at matched settings, averaged over environments: forgetting against not
        forgetting, which is the property the variant exists for and needs no cross-family tuning."""
        quick, never = [], []
        for seed in range(6):
            env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, seed + 1)
            if family == "ts_sw":
                a = dict(window=40, reward_sd=1.0, prior_sd=1.0)
                b = dict(window=100_000, reward_sd=1.0, prior_sd=1.0)
            else:
                a = dict(one_minus_gamma=0.05, reward_sd=1.0, prior_sd=1.0)
                b = dict(one_minus_gamma=1e-9, reward_sd=1.0, prior_sd=1.0)
            quick.append(BB.run_agent(env, family, a, 600, 50)["payoff"])
            never.append(BB.run_agent(env, family, b, 600, 50)["payoff"])
        assert np.mean(quick) > np.mean(never), f"{family}: {np.mean(quick):.4f} vs {np.mean(never):.4f}"


# ================================================================ C. change-detecting UCB
class TestChangeDetectingUCB:
    """Stage C: CUSUM-UCB checked against the published equations, and robust location under heavy tails."""

    ENV = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    CFG = dict(eps=0.2, h=8.0, alpha=0.05, c=0.5, M=20)

    # ------------------------------------------------------------ the borrowed algorithm
    def test_the_cusum_recursion_is_the_published_one(self):
        """s+ = y - u0 - eps, g+ = max(0, g+ + s+); s- = u0 - y - eps, g- = max(0, g- + s-); both floored at 0."""
        gp, gm = 0.0, 0.0
        ref, eps = 10.0, 0.5
        for y, want_p, want_m in [(10.0, 0.0, 0.0),            # on the reference: the slack absorbs it
                                  (12.0, 1.5, 0.0),            # above: the upper walk moves by y - u0 - eps
                                  (12.0, 3.0, 0.0),
                                  (6.0, 0.0, 3.5),             # below: upper floored at zero, lower moves
                                  (6.0, 0.0, 7.0)]:
            gp, gm = BB.cusum_step(gp, gm, y, ref, eps)
            assert gp == pytest.approx(want_p) and gm == pytest.approx(want_m)

    def test_neither_walk_can_go_negative(self):
        gp, gm = BB.cusum_step(0.0, 0.0, 10.0, 10.0, 5.0)
        assert gp == 0.0 and gm == 0.0

    def test_the_detector_alarms_on_a_shift_and_not_on_a_flat_stream(self):
        T = 600
        flat = np.zeros((T, 2))
        shifted = np.zeros((T, 2))
        shifted[T // 2:, :] = 6.0
        draws = BB.detector_draws(self.ENV, T)
        _, quiet = BB.cusum_ucb_actions(flat, 0.2, 3.0, 0.5, 0.5, 10, draws)
        _, loud = BB.cusum_ucb_actions(shifted, 0.2, 3.0, 0.5, 0.5, 10, draws)
        assert quiet.sum() == 0, "a constant stream raised a change alarm"
        assert loud.sum() > 0, "a step of six raised no alarm"
        assert np.flatnonzero(loud)[0] >= T // 2, "an alarm fired before the shift"

    def test_an_alarm_restarts_that_arm_rather_than_the_whole_agent(self):
        """After an alarm the arm discards its history and re-references at the new level, so a single step gives
        exactly one alarm and a second step is needed for a second. Between them it must take M fresh samples."""
        T = 500
        obs = np.zeros((T, 2))
        obs[100:250, 0] = 10.0                         # up, then back down: two changes to find
        draws = np.zeros((2, T))                       # no uniform exploration: deterministic
        _, one_step = BB.cusum_ucb_actions(np.where(np.arange(T)[:, None] >= 100, [10.0, 0.0], 0.0),
                                           0.2, 3.0, 0.0, 0.5, 10, draws)
        assert one_step.sum() == 1, "a single step should alarm once and then re-reference at the new level"
        _, two_steps = BB.cusum_ucb_actions(obs, 0.2, 3.0, 0.0, 0.5, 10, draws)
        fired = np.flatnonzero(two_steps)
        assert len(fired) >= 2, "two changes produced fewer than two alarms"
        assert int(np.diff(fired).min()) > 10, "an alarm fired before M fresh samples had been taken"

    def test_uniform_exploration_happens_at_the_stated_rate(self):
        T = 2000
        obs = np.zeros((T, 2))
        obs[:, 0] = 5.0                                 # arm 0 obviously better: any arm-1 play is exploration
        draws = BB.detector_draws(self.ENV, T)
        a_none, _ = BB.cusum_ucb_actions(obs, 0.2, 1e9, 0.0, 0.5, 20, draws)
        a_lots, _ = BB.cusum_ucb_actions(obs, 0.2, 1e9, 0.4, 0.5, 20, draws)
        assert (a_none[50:] == 1).mean() == 0.0, "an agent with alpha = 0 explored anyway"
        assert 0.1 < (a_lots[50:] == 1).mean() < 0.35, "the exploration rate is far from alpha / K"

    def test_only_the_chosen_option_is_observed(self):
        T = 400
        _, _, obs = BB._stream(self.ENV, T)
        draws = BB.detector_draws(self.ENV, T)
        run = lambda o: BB.cusum_ucb_actions(o, self.CFG["eps"], self.CFG["h"], self.CFG["alpha"], self.CFG["c"],
                                             self.CFG["M"], draws)[0]
        base = run(obs.copy())
        poisoned = obs.copy().astype(float)
        for t, at in enumerate(base):
            poisoned[t, 1 - int(at)] += 1000.0
        assert np.array_equal(run(poisoned), base)

    def test_the_adaptations_to_this_task_are_documented(self):
        f = BB.FAMILIES["cusum_ucb"]
        assert "liu2018" in f.sources and "liu2018" in REFERENCES
        for phrase in ("known", "bounded in [0, 1]", "tuned"):
            assert phrase in f.assumptions, f"the family does not record the assumption {phrase!r}"
        assert "adaptation" in BB.FAMILIES["cusum_ucb_robust"].assumptions

    # ------------------------------------------------------------ robust location
    def test_median_of_means_resists_an_outlier_that_moves_the_mean(self):
        x = np.r_[np.zeros(40), 1000.0]
        assert abs(float(x.mean())) > 20
        assert abs(BB.median_of_means(x, blocks=5)) < 1.0

    def test_median_of_means_agrees_with_the_mean_on_clean_symmetric_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, 4000)
        assert BB.median_of_means(x, blocks=5) == pytest.approx(float(x.mean()), abs=0.1)

    def test_median_of_means_falls_back_rather_than_inventing_precision(self):
        assert BB.median_of_means(np.array([3.0]), blocks=5) == pytest.approx(3.0)
        assert BB.median_of_means(np.array([]), blocks=5) == 0.0

    def test_the_robust_family_differs_only_in_the_location_estimator(self):
        a = BB.actions(self.ENV, "cusum_ucb", self.CFG, 400, "heavy")
        b = BB.actions(self.ENV, "cusum_ucb_robust", self.CFG, 400, "heavy")
        assert not np.array_equal(a, b)
        assert set(BB.FAMILIES["cusum_ucb"].space) == set(BB.FAMILIES["cusum_ucb_robust"].space)

    # ------------------------------------------------------------ the heavy-tailed condition
    def test_the_heavy_condition_is_actually_heavy_tailed(self):
        kurt = lambda o: float(((o - o.mean()) ** 4).mean() / ((o - o.mean()) ** 2).mean() ** 2)
        light = kurt(BB._stream(self.ENV, 2000)[2])
        heavy = kurt(BB._stream(self.ENV, 2000, "heavy")[2])
        assert 2.0 < light < 4.0, f"the default condition is not approximately Gaussian: kurtosis {light:.1f}"
        assert heavy > 20.0, f"the heavy condition has kurtosis {heavy:.1f}, which is not heavy-tailed"

    def test_the_heavy_condition_changes_only_what_is_observed(self):
        """Agents are still scored on the truth; only their evidence is contaminated."""
        s_g, o_g, obs_g = BB._stream(self.ENV, 500)
        s_h, o_h, obs_h = BB._stream(self.ENV, 500, "heavy")
        assert np.array_equal(s_g, s_h) and np.array_equal(o_g, o_h)
        assert not np.array_equal(obs_g, obs_h)

    def test_every_family_faces_the_same_condition(self):
        """Identical information restrictions: the condition belongs to the run, not to the agent."""
        assert np.array_equal(BB._stream(self.ENV, 300, "heavy")[2], BB._stream(self.ENV, 300, "heavy")[2])
        for fam, cfg in (("cusum_ucb", self.CFG), ("ts_sw", dict(window=50, reward_sd=1.0, prior_sd=1.0))):
            assert np.isfinite(BB.run_agent(self.ENV, fam, cfg, 300, 50, "heavy")["payoff"])

    def test_an_unknown_condition_is_refused(self):
        with pytest.raises(ValueError):
            BB._stream(self.ENV, 100, "lognormalish")

    def test_robust_location_does_not_pay_at_these_untuned_settings(self):
        """Measured, not assumed. Resistance to outliers is a property of the estimator and is tested directly
        above; whether substituting it PAYS in this environment is an empirical question, and at one shared
        untuned configuration averaged over six environments it does not. Recorded here so the claim is not
        quietly made elsewhere: the tuned comparison is the place to settle it, since the two families have the
        same parameters and may want different values for them.
        """
        def mean_payoff(family, tail):
            return float(np.mean([BB.run_agent(tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, s + 1),
                                               family, self.CFG, 800, 50, tail)["payoff"] for s in range(6)]))
        robust, plain = mean_payoff("cusum_ucb_robust", "heavy"), mean_payoff("cusum_ucb", "heavy")
        assert np.isfinite(robust) and np.isfinite(plain)
        assert robust < plain, ("robust location now pays at these settings; the finding recorded in this test "
                                "has changed and the documentation that cites it needs updating")


# ================================================================ D. moment-based distributional robustness
class TestMomentDRO:
    """Stage D: an ambiguity set over distributions with given moments, which can price unobserved outcomes."""

    ENV = tasks.TaskEnv("inventory", 20.0, 0.01, 10.0, 0.1, 4)

    # ------------------------------------------------------------ the closed form, checked not trusted
    def test_the_closed_form_minimizes_the_worst_case_loss_it_claims_to(self):
        """Scarf's order against a grid search over the worst-case objective. If the algebra were wrong, the
        closed form and the numerical minimizer would part company here."""
        for mu, sd in [(100.0, 20.0), (50.0, 5.0), (120.0, 45.0)]:
            S = BI.scarf_order(mu, sd)
            grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 40001)
            numerical = grid[int(np.argmin([BI.scarf_worst_case_loss(x, mu, sd) for x in grid]))]
            assert S == pytest.approx(float(numerical), abs=1e-2)

    def test_the_order_is_the_published_expression(self):
        """S* = mu + (sigma / 2) (sqrt(p / h) - sqrt(h / p))."""
        mu, sd = 100.0, 20.0
        want = mu + 0.5 * sd * (np.sqrt(BI.P_SHORT / BI.H_OVER) - np.sqrt(BI.H_OVER / BI.P_SHORT))
        assert BI.scarf_order(mu, sd) == pytest.approx(want)

    def test_with_no_estimated_variation_it_orders_the_mean(self):
        assert BI.scarf_order(80.0, 0.0) == pytest.approx(80.0)

    def test_the_order_is_never_negative(self):
        assert BI.scarf_order(1.0, 0.0) >= 0.0
        assert BI.scarf_order(-50.0, 1.0) == 0.0

    def test_it_is_more_cautious_than_the_normal_quantile_at_this_cost_ratio(self):
        """Distribution-free is not the same as conservative in every direction: at a 0.8 critical fractile the
        moment bound orders LESS than the Gaussian quantile, because the worst case it guards against is a
        two-point distribution rather than a normal tail. Recorded so the direction is not assumed."""
        mu, sd = 100.0, 20.0
        assert BI.scarf_order(mu, sd) < mu + BI.Z_FRAC * sd

    # ------------------------------------------------------------ what separates it from the empirical model
    def test_the_empirical_support_model_cannot_leave_the_observed_range(self):
        window = np.array([[90.0, 110.0]])
        empirical = float(BI.dro_orders_window(window, 0.05)[0])
        assert 90.0 - 1e-9 <= empirical <= 110.0 + 1e-9

    def test_the_moment_model_can_order_outside_the_observed_range(self):
        """The decisive difference. The worst case is attained by a two-point distribution that need not sit on
        observed values, so an outcome never seen can still be priced."""
        window = np.array([90.0, 110.0])
        moment = BI.moment_order_window(window)
        assert moment > window.max(), f"the moment order {moment:.2f} stayed inside the observed range"

    def test_the_moment_model_depends_on_the_window_only_through_its_first_two_moments(self):
        """Two windows with the same mean and s.d. but different shapes: the moment model cannot tell them apart,
        the empirical-support model can. That is what 'moment ambiguity' means, stated as a test."""
        a = np.array([80.0, 80.0, 120.0, 120.0])
        # same mean and same sample s.d., different shape: two-point against a spread-out quartet
        b = np.array([100.0 - 20.0 * np.sqrt(2), 100.0, 100.0, 100.0 + 20.0 * np.sqrt(2)])
        assert float(a.mean()) == pytest.approx(float(b.mean()))
        assert float(a.std(ddof=1)) == pytest.approx(float(b.std(ddof=1)))
        assert BI.moment_order_window(a) == pytest.approx(BI.moment_order_window(b))
        emp_a = float(BI.dro_orders_window(a[None, :], 0.05)[0])
        emp_b = float(BI.dro_orders_window(b[None, :], 0.05)[0])
        assert emp_a != pytest.approx(emp_b), "the empirical-support model ignored the shape of its window"

    # ------------------------------------------------------------ naming and estimation
    def test_the_existing_model_is_named_for_what_it_actually_assumes(self):
        name = BI.FAMILIES["dro"].name
        assert "empirical support only" in name
        assert "cannot place mass on unseen values" in BI.FAMILIES["dro"].assumptions

    def test_the_moment_family_says_what_its_ambiguity_set_is(self):
        f = BI.FAMILIES["dro_moment"]
        assert f.role == "competitor"
        assert "SUPPORT" in f.assumptions and "never given" in f.assumptions
        for key in ("scarf1958", "gallego1993"):
            assert key in f.sources and key in REFERENCES

    def test_the_spread_multiplier_is_tuned_rather_than_the_true_measurement_error(self):
        env = self.ENV
        a = BI.orders(env, "dro_moment", dict(window=40, sd_scale=0.4), 300)
        b = BI.orders(env, "dro_moment", dict(window=40, sd_scale=1.8), 300)
        assert not np.allclose(a, b), "sd_scale had no effect, so it is not being used"
        assert float(np.mean(b[60:])) > float(np.mean(a[60:])), "a wider assumed spread must not order less"
        assert set(BI.FAMILIES["dro_moment"].space) == {"window", "sd_scale"}

    def test_it_runs_over_a_whole_environment_and_warms_up_on_the_default(self):
        S = BI.orders(self.ENV, "dro_moment", dict(window=40, sd_scale=1.0), 300)
        assert len(S) == 300 and np.isfinite(S).all() and (S >= 0).all()

    # ------------------------------------------------------------ the registered study is untouched
    def test_the_registered_learnability_plan_does_not_move(self):
        """The learnability plan digests six named objects from this module. Stage D adds functions and renames a
        display label; it must not touch those."""
        from heiner_abm import learnability as L
        from heiner_abm import registered as R
        assert L.LearnPlan().digest == R.LEARN_PLAN
