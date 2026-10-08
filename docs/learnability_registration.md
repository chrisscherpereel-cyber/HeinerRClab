# Registration: when can reliability be learned before the environment changes?

* Plan hash: `8f43bedc7dae10bc` (registered in the repository: `8f43bedc7dae10bc`)
* Status: This specification is frozen in the repository: its hash covers the plan and the study code, and a test fails if either changes without a new registration entry. It has **not** been preregistered with an external registry (for example OSF or AsPredicted). Any external registration must be done separately; until then, do not describe the study as externally preregistered.

## Question
When does an agent have enough informative feedback within a regime to learn whether adapting is reliable, before the environment changes? A proposed construct, the learnability ratio R (observations needed to determine the sign of the advantage of adapting with 90% one-sided confidence, divided by the informative observations available within a regime), is tested for whether it explains policy performance beyond volatility and observation noise.

## Design
A newsvendor (Arrow, Harris & Marschak 1951): each period the agent orders S before demand d is known; leftover
    stock costs H_OVER = 1 and shortage P_SHORT = 4 per unit, so with a known demand distribution the best order is
    its 0.8-quantile. The mean demand mu_t changes over time (process families below); d_t = max(0, mu_t + sigma e_t);
    the agent observes y_t = d_t + tau u_t, but only with probability `avail` (otherwise the period's demand is never
    observed). Orders use observations up to t - 1.

Manipulated factors (Config)
    sigma     outcome volatility (s.d. of demand around its mean)
    tau       observation error
    hazard    regime-change frequency
    avail     feedback availability (probability that a period's demand is observed)
    memory    the reliability gates' memory lambda (their effective window is about (1 + lambda) / (1 - lambda))
    default   default quality: 'slow' (slow forecast, gain 0.01), 'fixed_initial' (the best order for the first
              regime, which deteriorates after the first change), 'biased' (40 units too low), 'dominated' (orders 0)
    cost      adaptation cost c, charged in every period whose order differs from the default's

Process families (Config.family): every family starts at mu = 100 and keeps mu in [30, 170]
    jump      with probability `hazard` the mean jumps by N(0, 30^2) (tuning and the main test)
    switch    with probability `hazard` the mean switches between 70 and 130 (new family, test only)
    drift     the mean follows a random walk with per-period s.d. 30 sqrt(hazard), the same variance per expected regime
              length as `jump`, without discrete regimes (new family, test only)

Policies (all see the same exogenous paths: common random numbers)
    always      always adapt: order the fast forecast's 0.8-quantile, S_F = forecast + z * sd (gain tuned)
    default     retain the default S_D
    band        inaction band: S_F if |S_F - S_D| > b * sd, else S_D (b tuned)
    gate_gain   existing estimated-gain gate (heiner_abm.gates): per size bin of |S_F - S_D| / sd, adapt if the
                learned mean of g = payoff(S_F, y) - payoff(S_D, y) is at least c
    gate_lcb    confidence-sensitive gate: adapt if the bin has at least `nmin` effective observations and the
                one-sided lower bound at confidence `conf` exceeds c; (conf, nmin) are tuned on the training
                configurations with the same budget as every other tunable policy (both gates learn from every
                observed period, whatever they chose: the observed demand values both orders; feedback is released
                at the end of the period)
    bocpd       Bayesian change detection (Adams & MacKay 2007), misspecified Gaussian reset model, tuned; orders its
                predictive 0.8-quantile; it does not use the default
    dro         distributionally robust newsvendor on the last N observed demands (modified chi-squared ball,
                Ben-Tal et al. 2013), N and rho tuned
    oracle      PERFECT INFORMATION (knows mu_t): order max(0, mu_t + z sigma); a bound, not ranked
Every policy pays c whenever its order differs from the default's.

Equal tuning budget
    Every policy with free hyperparameters gets the same number of candidate settings (len of its entry in
    LearnPlan.tuning_grid, four by default), each scored on the same training configurations and paths, and the best
    is carried into the pilot and test runs: `always` (forecast gain), `band` (b), `gate_lcb` (conf, nmin), `bocpd`
    (hazard, prior sd, obs sd) and `dro` (window, rho). `default` and `gate_gain` have no free hyperparameters, so
    they receive none; `oracle` is a bound, not a competitor. tune() returns a log with one row per policy giving the
    number of candidates evaluated, so the equality is checkable rather than asserted. Before 7 October 2026 the two
    gates received no tuning at all while their four comparators received four candidates each, and LearnPlan's
    `gate_confidence` / `gate_min_evidence` were never read: see LearnPlan.tune_gate for the legacy setting.

Outcomes (per path, recorded periods t >= burn_in)
    primary     net payoff per period = -(H_OVER (S - d)+ + P_SHORT (d - S)+) - c 1[S != S_D]
    secondary   regret (oracle's gross payoff minus net payoff); downside loss (mean of the worst 5% of periods' net
                payoff, CVaR 5%); calibration of the gates' predicted advantage (slope and bias against the realized
                observed gain); adaptation rate (share of periods with S != S_D); missed opportunities (share of
                periods whose true gain of S_F over S_D exceeds c in which the policy did not order S_F; policies that
                choose between S_D and S_F only); recovery delay (periods after a change until the policy's 20-period
                rolling excess loss over the oracle falls to its own median outside the 50 periods after changes;
                censored at the next change)

## Proposed construct
the learnability ratio R (measured on independent pilot paths, never on test paths)
    For each regime of a pilot path in which the two candidate orders differ (for `drift`, consecutive blocks of the
    nominal length 1 / hazard):
        mu_r     the regime's true mean advantage of S_F over S_D (from the pilot's true demands: researcher data)
        s        the noise of what the learner observes: s.d. of observed gains around their regime's mu_r, pooled
        n_needed = (z_0.9 s / |mu_r|)^2   observations to determine the sign of the regime's advantage with 90%
                                           one-sided confidence (the specified precision), clipped to [1, 1e6]
        n_avail  = informative observations in the regime (demand observed and S_F != S_D)
    R = median over regimes of n_needed / max(n_avail, 1). R < 1: the sign of the advantage of adapting can typically
    be learned within a regime; R > 1: it cannot. R is a proposed construct: whether it explains performance beyond
    volatility and observation noise is a hypothesis of the study (H2), not an assumption.

## Primary outcome
Net payoff per period (newsvendor payoff minus the adaptation cost c whenever the order differs from the default's). Effects are paired differences over shared exogenous paths with 95% bootstrap intervals; rankings are not used to declare any theory superior.

## Hypotheses and decision rules
* **H1.** Learnability: the confidence-sensitive gate's advantage over the better fixed rule (always adapt or retain the default, whichever was better on pilot paths) falls as the learnability ratio R rises. *Decision rule:* Across the jump-family test configurations, OLS slope of the configuration-mean advantage (net payoff per period) on log10 R; 95% bootstrap CI over configurations. Supported if the CI lies below 0.
* **H2.** Incremental validity of R: log10 R explains the gate's advantage beyond outcome volatility and observation noise. *Decision rule:* Leave-one-configuration-out cross-validated R² of advantage ~ log sigma + log(1 + tau) + log10 R minus that of advantage ~ log sigma + log(1 + tau); 95% bootstrap CI over configurations. Supported if the CI lies above 0.
* **H3.** Transfer to new process families: the H1 relation holds when R is measured and the gate is tested on the switching and drifting families, which were not used for tuning. *Decision rule:* OLS slope of the advantage on log10 R pooled over the two new families, 95% bootstrap CI over configurations. Supported if the CI lies below 0.
* **H4.** Replication in the market: the same relation holds for the market's confidence-sensitive gate (heiner_abm.learnability_market). *Decision rule:* As H1, in the market configurations. Supported if the 95% bootstrap CI of the slope lies below 0.

## Negative controls
* **NC1 · Perfect information and reversible, costless adaptation.** The flexible order is the oracle's order and c = 0. Expected: no policy beats always adapting. Passes if no policy's paired net-payoff difference over always adapting has a 95% CI lower bound above 0.25 per period.
* **NC2 · Dominated default.** The default orders nothing. Expected: the gates learn to adapt. Passes if both gates adapt in at least 90% of periods and lose no more than 5% of always adapting's net payoff.
* **NC3 · Stable environment with abundant feedback.** No regime changes, every demand observed, little noise, 3,000 periods. Expected: R < 1 and the confidence-sensitive gate is at least as good as the better fixed rule. Passes if R < 1 and the gate's advantage has a 95% CI upper bound above −0.25 per period.

## Sets
Training configurations (jump family) for tuning; independent pilot paths for R and for choosing the better fixed rule; untouched test paths (new seeds) for every reported effect; new process families (switching, drifting) for transfer. Seed blocks are disjoint by construction.

## Plan
```json
{
 "baseline": [
  [
   "family",
   "jump"
  ],
  [
   "sigma",
   20.0
  ],
  [
   "tau",
   10.0
  ],
  [
   "hazard",
   0.01
  ],
  [
   "avail",
   1.0
  ],
  [
   "memory",
   0.97
  ],
  [
   "default",
   "fixed_initial"
  ],
  [
   "cost",
   1.0
  ],
  [
   "perfect",
   false
  ]
 ],
 "burn_in": 100,
 "code": "d984038f495b6ade82225a36c8815aabd7411c416043a8a1ff745db83e3e8b7b",
 "gate_confidence": 0.9,
 "gate_min_evidence": 5.0,
 "hypotheses": [
  [
   "H1",
   "Learnability: the confidence-sensitive gate's advantage over the better fixed rule (always adapt or retain the default, whichever was better on pilot paths) falls as the learnability ratio R rises.",
   "Across the jump-family test configurations, OLS slope of the configuration-mean advantage (net payoff per period) on log10 R; 95% bootstrap CI over configurations. Supported if the CI lies below 0."
  ],
  [
   "H2",
   "Incremental validity of R: log10 R explains the gate's advantage beyond outcome volatility and observation noise.",
   "Leave-one-configuration-out cross-validated R\u00b2 of advantage ~ log sigma + log(1 + tau) + log10 R minus that of advantage ~ log sigma + log(1 + tau); 95% bootstrap CI over configurations. Supported if the CI lies above 0."
  ],
  [
   "H3",
   "Transfer to new process families: the H1 relation holds when R is measured and the gate is tested on the switching and drifting families, which were not used for tuning.",
   "OLS slope of the advantage on log10 R pooled over the two new families, 95% bootstrap CI over configurations. Supported if the CI lies below 0."
  ],
  [
   "H4",
   "Replication in the market: the same relation holds for the market's confidence-sensitive gate (heiner_abm.learnability_market).",
   "As H1, in the market configurations. Supported if the 95% bootstrap CI of the slope lies below 0."
  ]
 ],
 "lhs_ranges": [
  [
   "sigma",
   [
    5.0,
    40.0
   ]
  ],
  [
   "tau",
   [
    0.0,
    30.0
   ]
  ],
  [
   "hazard",
   [
    0.002,
    0.05
   ]
  ],
  [
   "avail",
   [
    0.2,
    1.0
   ]
  ],
  [
   "memory",
   [
    0.9,
    0.995
   ]
  ]
 ],
 "n_boot": 2000,
 "n_new_family_configs": 15,
 "n_test_configs": 30,
 "n_train_configs": 8,
 "negative_controls": [
  [
   "NC1",
   "Perfect information and reversible, costless adaptation",
   "The flexible order is the oracle's order and c = 0. Expected: no policy beats always adapting. Passes if no policy's paired net-payoff difference over always adapting has a 95% CI lower bound above 0.25 per period."
  ],
  [
   "NC2",
   "Dominated default",
   "The default orders nothing. Expected: the gates learn to adapt. Passes if both gates adapt in at least 90% of periods and lose no more than 5% of always adapting's net payoff."
  ],
  [
   "NC3",
   "Stable environment with abundant feedback",
   "No regime changes, every demand observed, little noise, 3,000 periods. Expected: R < 1 and the confidence-sensitive gate is at least as good as the better fixed rule. Passes if R < 1 and the gate's advantage has a 95% CI upper bound above \u22120.25 per period."
  ]
 ],
 "new_families": [
  "switch",
  "drift"
 ],
 "paths": 10,
 "periods": 1500,
 "pilot_paths": 6,
 "seed": 20261007,
 "sweeps": [
  [
   "sigma",
   [
    5.0,
    20.0,
    40.0
   ]
  ],
  [
   "tau",
   [
    0.0,
    10.0,
    30.0
   ]
  ],
  [
   "hazard",
   [
    0.002,
    0.01,
    0.05
   ]
  ],
  [
   "avail",
   [
    1.0,
    0.5,
    0.2
   ]
  ],
  [
   "memory",
   [
    0.9,
    0.97,
    0.995
   ]
  ],
  [
   "default",
   [
    "slow",
    "fixed_initial",
    "biased"
   ]
  ]
 ],
 "train_paths": 3,
 "tune_gate": true,
 "tuning_grid": [
  [
   "gain",
   [
    0.05,
    0.1,
    0.2,
    0.4
   ]
  ],
  [
   "band_b",
   [
    0.25,
    0.5,
    1.0,
    2.0
   ]
  ],
  [
   "gate",
   [
    [
     0.9,
     5.0
    ],
    [
     0.8,
     3.0
    ],
    [
     0.9,
     12.0
    ],
    [
     0.95,
     25.0
    ]
   ]
  ],
  [
   "bocpd",
   [
    [
     0.005,
     30.0,
     25.0
    ],
    [
     0.01,
     40.0,
     25.0
    ],
    [
     0.02,
     40.0,
     20.0
    ],
    [
     0.005,
     60.0,
     35.0
    ]
   ]
  ],
  [
   "dro",
   [
    [
     20,
     0.01
    ],
    [
     40,
     0.01
    ],
    [
     40,
     0.1
    ],
    [
     80,
     0.05
    ]
   ]
  ]
 ],
 "version": "1.0"
}
```
