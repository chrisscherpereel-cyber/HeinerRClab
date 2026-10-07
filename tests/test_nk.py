"""NK-landscape environment (heiner_abm.nk): K_NK = 0 separability, seeded reproducibility, exact small-case payoffs,
topologies, benchmarks, identical proposals for the gates and information restrictions."""
import itertools
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm import nk


def test_k_nk_bounds_and_topologies():
    with pytest.raises(ValueError):
        nk.interactions(8, 8, "adjacent")
    with pytest.raises(ValueError):
        nk.interactions(8, -1, "adjacent")
    for top in nk.TOPOLOGIES:
        for K in (0, 2, 4, 7):
            nbr = nk.interactions(8, K, top, np.random.default_rng(1))
            assert nbr.shape == (8, K + 1)
            assert all(nbr[i, 0] == i and len(set(nbr[i])) == K + 1 for i in range(8))
    assert nk.interactions(6, 2, "block").tolist() == [[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 4, 5], [4, 3, 5], [5, 3, 4]]
    assert nk.interactions(5, 2, "block")[3].tolist() == [3, 4, 0]           # the last block wraps to the start
    explicit = ((1,), (0,), (0,))
    assert nk.interactions(3, 1, explicit).tolist() == [[0, 1], [1, 0], [2, 0]]
    with pytest.raises(ValueError):
        nk.interactions(3, 1, ((0,), (0,), (0,)))                              # a component cannot be its own partner


def test_exact_small_case_payoffs():
    # N = 3, K_NK = 1, adjacent: C_i depends on (x_i, x_{i+1 mod 3}); table row = x_i + 2 * x_{i+1}
    tables = np.array([[0.1, 0.9, 0.4, 0.2], [0.5, 0.3, 0.8, 0.6], [0.7, 0.0, 1.0, 0.25]])
    land = nk.Landscape(nk.interactions(3, 1, "adjacent"), tables)
    for x in range(8):
        b = [(x >> i) & 1 for i in range(3)]
        by_hand = np.mean([tables[i, b[i] + 2 * b[(i + 1) % 3]] for i in range(3)])
        assert land.fitness([x])[0] == pytest.approx(by_hand)
    F, peak = nk._enumerate(land)
    assert F.max() == pytest.approx(max(land.fitness([x])[0] for x in range(8)))
    for x in range(8):
        assert peak[x] == all(F[x] >= F[x ^ (1 << i)] for i in range(3))


def test_k_nk_zero_is_separable():
    env = nk.NKEnv(N=8, K_NK=0, seed=5)
    ep = nk.landscape_path(env)[0]
    F = ep.F
    for i in range(8):                         # the effect of flipping component i does not depend on the others
        d = np.array([F[x ^ (1 << i)] - F[x] for x in range(256) if not (x >> i) & 1])
        assert np.allclose(d, d[0])
    best = sum(1 << i for i in range(8) if ep.land.tables[i, 1] > ep.land.tables[i, 0])
    assert F[best] == pytest.approx(F.max()) and ep.peak.sum() == 1            # a single peak: the global maximum
    clean = replace(env, obs_noise=0.0, budget=100, periods=120, cost=0.0)
    df = nk.run_landscape(clean, [1, 2, 3], searchers=("hill",))
    assert (df["stuck_share"] == 0).all() and (df["false_improvements"] == 0).all()


def test_seeded_reproducibility():
    env = nk.NKEnv(K_NK=4, hazard=0.02, obs_noise=0.03, topology="random", seed=11, periods=200, budget=120)
    a, b = nk.landscape_path(env), nk.landscape_path(env)
    assert [e.start for e in a] == [e.start for e in b] and len(a) > 1
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.land.tables, y.land.tables)
        np.testing.assert_array_equal(x.land.nbr, y.land.nbr)
    r1, r2 = nk.run_landscape(env, [1, 2]), nk.run_landscape(env, [1, 2])
    assert r1.equals(r2)
    other = nk.landscape_path(replace(env, seed=12))
    assert not np.array_equal(other[0].land.tables, a[0].land.tables)
    # noise and change are separate streams: changing the noise level leaves the landscape path untouched
    c = nk.landscape_path(replace(env, obs_noise=0.2))
    assert [e.start for e in c] == [e.start for e in a]


def test_changes_redraw_only_the_stated_share_of_components():
    env = nk.NKEnv(K_NK=2, hazard=0.05, change_frac=0.25, seed=3, periods=200)
    path = nk.landscape_path(env)
    for prev, cur in zip(path[:-1], path[1:]):
        changed = (prev.land.tables != cur.land.tables).any(1).sum()
        assert changed == 2 and np.array_equal(prev.land.nbr, cur.land.nbr)


def test_benchmark_exact_for_small_and_best_known_for_large():
    small = nk.run_landscape(nk.NKEnv(N=8, K_NK=2, seed=1, periods=60, budget=40), [1], searchers=("hill",))
    assert set(small["benchmark"]) == {"exact"}
    large = nk.run_landscape(nk.NKEnv(N=nk.ENUM_MAX + 2, K_NK=3, seed=1, periods=40, budget=30), [1],
                             searchers=("hill", "stochastic"))
    assert set(large["benchmark"]) == {"best-known"} and (large["regret"] >= -1e-12).all()


def _streams(env, seed=1):
    path = nk.landscape_path(env)
    st = nk.run_streams(env, path, seed)
    return path, st, nk.Observer(env, path, st.noise, st.leader, st.visible)


def test_gates_get_identical_proposals_and_budgets():
    env = nk.NKEnv(K_NK=4, obs_noise=0.05, seed=4, periods=300, budget=200)
    path, st, obs = _streams(env)
    logs, used = {}, {}
    for g in nk.GATED:
        rec = {}
        logs[g] = nk.simulate(g, env, obs, st.start, st.U, nk.SearchParams(), rec)[2]
        used[g] = rec["used"]
    assert all(u <= env.budget for u in used.values())
    a, b = logs["gate_none"], logs["gate_lcb"]
    for da, db in zip(a, b):                     # same proposals until the gates' first different decision
        assert (da["t"], da["x"], da["y"]) == (db["t"], db["x"], db["y"])
        if da["accept"] != db["accept"]:
            break
    else:
        pytest.fail("the gates never decided differently in this case")


class _Recorder(nk.Observer):
    def __init__(self, inner):
        self.inner, self.seen = inner, {}

    def observe(self, x, t, slot):
        v = self.inner.observe(x, t, slot)
        self.seen[(x, t, slot)] = v
        return v

    def leader(self, t):
        v = self.inner.leader(t)
        self.seen[("leader", t)] = v
        return v


class _Replay(nk.Observer):
    """Has no landscape at all: only the observations recorded from a run."""

    def __init__(self, seen):
        self.seen = seen

    def observe(self, x, t, slot):
        return self.seen[(x, t, slot)]

    def leader(self, t):
        return self.seen[("leader", t)]


@pytest.mark.parametrize("searcher", nk.SEARCHERS)
def test_decisions_depend_only_on_observations(searcher):
    env = nk.NKEnv(K_NK=4, obs_noise=0.03, hazard=0.02, seed=6, periods=200, budget=150)
    path, st, obs = _streams(env)
    rec = _Recorder(obs)
    xs, _, _ = nk.simulate(searcher, env, rec, st.start, st.U, nk.SearchParams())
    xs2, _, _ = nk.simulate(searcher, env, _Replay(rec.seen), st.start, st.U, nk.SearchParams())
    np.testing.assert_array_equal(xs, xs2)


@pytest.mark.parametrize("searcher", nk.SEARCHERS)
def test_no_lookahead(searcher):
    """Decisions up to the end of period K do not depend on observations or landscape changes after K."""
    env = nk.NKEnv(K_NK=4, obs_noise=0.03, hazard=0.0, seed=8, periods=200, budget=150)
    path, st, obs = _streams(env)
    K = 90
    noise2 = st.noise.copy()
    noise2[K + 1:] += 3.0
    path2 = path + [nk.Epoch(K + 1, nk.Landscape(path[0].land.nbr, np.random.default_rng(0).random(
        path[0].land.tables.shape)), None, 1.0, False)]
    obs2 = nk.Observer(env, path2, noise2, st.leader, st.visible)
    a = nk.simulate(searcher, env, obs, st.start, st.U, nk.SearchParams())[0]
    b = nk.simulate(searcher, env, obs2, st.start, st.U, nk.SearchParams())[0]
    np.testing.assert_array_equal(a[:K + 2], b[:K + 2])     # the configuration in K + 1 is decided at the end of K


@pytest.mark.parametrize("searcher", nk.GATED)
def test_verification_is_released_after_the_decision(searcher):
    env = nk.NKEnv(K_NK=4, obs_noise=0.05, seed=9, periods=200, budget=200)
    path, st, obs = _streams(env)
    a_log = nk.simulate(searcher, env, obs, st.start, st.U, nk.SearchParams())[2]
    K = next(d["t"] for d in a_log if np.isfinite(d["g2"]))
    noise2 = st.noise.copy()
    noise2[K, 2:] += 5.0                                    # change only period K's verification draws
    obs2 = nk.Observer(env, path, noise2, st.leader, st.visible)
    b_log = nk.simulate(searcher, env, obs2, st.start, st.U, nk.SearchParams())[2]
    da = next(d for d in a_log if d["t"] == K)
    db = next(d for d in b_log if d["t"] == K)
    assert da["accept"] == db["accept"] and da["g2"] != db["g2"]


def test_imitation_sees_only_observable_components():
    env = nk.NKEnv(K_NK=2, obs_noise=0.02, seed=10, periods=200, budget=150, observe=0.5)
    path, st, obs = _streams(env)
    hidden = sum(1 << i for i in range(env.N) if i not in set(st.visible.tolist()))
    assert len(st.visible) == 4
    obs2 = nk.Observer(env, path, st.noise, st.leader ^ hidden, st.visible)
    a = nk.simulate("imitate", env, obs, st.start, st.U, nk.SearchParams())[0]
    b = nk.simulate("imitate", env, obs2, st.start, st.U, nk.SearchParams())[0]
    np.testing.assert_array_equal(a, b)
    assert set(obs.leader(5)) == set(st.visible.tolist())
    none = nk.Observer(env, path, st.noise, st.leader, np.array([], dtype=int))
    assert none.leader(5) == {}


def test_noise_free_costless_hill_climbing_never_makes_false_improvements():
    env = nk.NKEnv(K_NK=4, obs_noise=0.0, cost=0.0, seed=2, periods=150, budget=150)
    df = nk.run_landscape(env, [1, 2, 3], searchers=("hill", "gate_none"))
    assert (df["false_improvements"] == 0).all() and (df["missed_improvements"] == 0).all()


def test_quick_study_and_summaries():
    plan = nk.NKPlan(K_values=(0, 4), noise_values=(0.0, 0.05), hazard_values=(0.0, 0.02), n_landscapes=3,
                     n_starts=2, periods=120, budget=80, n_boot=100)
    runs = nk.run_nk(plan)
    assert set(runs["searcher"]) == set(nk.SEARCHERS) and set(runs["K_NK"]) == {0, 4}
    s = nk.summarize(runs[runs["sweep"] == "noise"], "payoff", ["K_NK", "obs_noise", "searcher"], 100)
    assert (s["lo"] <= s["mean"] + 1e-12).all() and (s["mean"] <= s["hi"] + 1e-12).all()
    assert (s["landscapes"] == 3).all()
    d = nk.paired(runs, "gate_lcb", "gate_none", "payoff", ["K_NK"], 100)
    assert len(d) == 2
