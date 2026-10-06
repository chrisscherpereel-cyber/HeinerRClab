"""The encompassing test is robust to heavy tails and clipping in the forecasts it combines.

Every feature is mapped onto its training fold's empirical distribution before the logit is fitted, so the results
depend only on the features' ordering; a single-feature row is scored by the raw forecast in the direction learned on
the training folds, so it scores exactly like the forecast."""
import numpy as np
import pandas as pd

from heiner_abm.analysis import auc
from heiner_abm.experiments import _train_ranks, encompassing_test


def _synthetic(n_env=60, per_env=8, seed=3):
    """Firms whose outcome depends weakly on a heavy-tailed RC margin, with mass at the ±10 clipping bounds."""
    rng = np.random.default_rng(seed)
    n = n_env * per_env
    env = np.repeat(np.arange(n_env), per_env)
    latent = rng.standard_normal(n)
    ratio = np.exp(np.clip(latent * 4 + rng.standard_t(1.5, n), -12, 12))       # ln(ratio) heavy-tailed, clipped
    tol = np.exp(np.clip(rng.standard_t(1.5, n) * 3, -12, 12))
    rc = np.clip(np.log(ratio), -10, 10) - np.clip(np.log(tol), -10, 10)
    y = (0.6 * np.tanh(rc / 4) + rng.standard_normal(n)) > 0
    return pd.DataFrame({"env": env, "ratio_est_full": ratio, "tol_est_full": tol,
                         "K_est": rng.gamma(2.0, 1.0, n), "delta": rng.uniform(2, 30, n),
                         "instability": rng.uniform(0, 1.5, n), "dyn_adv_est": rng.standard_normal(n) + 0.3 * y,
                         "dyn_adv_eval": np.where(y, 1.0, -1.0)})


def _single_row(df, **kw):
    enc, _ = encompassing_test(df, n_boot=20, **kw)
    return float(enc.loc[enc["model"] == "RC margin alone", "cv_auc"].iloc[0])


def test_train_ranks_are_the_training_cdf():
    train = np.array([3.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(_train_ranks(train, np.array([0.0, 1.0, 2.0, 2.5, 5.0])), [0.0, 0.125, 0.5, 0.75, 1.0])


def test_single_feature_row_scores_like_the_raw_feature():
    df = _synthetic()
    row = _single_row(df)
    rc = np.clip(np.log(df["ratio_est_full"]), -10, 10) - np.clip(np.log(df["tol_est_full"]), -10, 10)
    y = df["dyn_adv_eval"].to_numpy() > 0
    # same environment folds as encompassing_test (seed 0, 5 folds)
    uniq = np.unique(df["env"])
    fold_of = dict(zip(np.random.default_rng(0).permutation(uniq), np.arange(len(uniq)) % 5))
    fid = np.array([fold_of[e] for e in df["env"]])
    raw = np.average([auc(rc[fid == f], y[fid == f]) for f in range(5)], weights=[np.sum(fid == f) for f in range(5)])
    assert abs(row - raw) < 1e-9


def test_results_depend_only_on_the_ordering_of_each_forecast():
    df = _synthetic()
    enc, gain = encompassing_test(df, n_boot=20)
    df2 = df.assign(delta=np.exp(df["delta"] / 3), K_est=df["K_est"] ** 3, dyn_adv_est=np.sinh(df["dyn_adv_est"]))
    enc2, gain2 = encompassing_test(df2, n_boot=20)
    np.testing.assert_allclose(enc["cv_auc"], enc2["cv_auc"])
    assert gain["gain"] == gain2["gain"]


def test_single_feature_row_is_stable_when_one_environment_is_left_out():
    df = _synthetic()
    vals = [_single_row(df[df["env"] != e]) for e in range(0, 60, 6)]
    assert max(vals) - min(vals) < 0.03
