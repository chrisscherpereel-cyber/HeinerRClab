# Registration: when can reliability be learned before the environment changes?

* Plan hash: `a98b971b49fe5e3e` (registered in the repository: `a98b971b49fe5e3e`)
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
    cost      default-departure overhead: charged in every period whose order differs from the default's
    switch_cost     fixed switching cost: charged in every period whose order differs from the PREVIOUS period's
    magnitude_cost  charged per unit of |S_t - S_{t-1}|
    cost_model      "components" (all three above) or "legacy_departure_only" (see below)

Three distinct costs, deliberately not interchangeable
    Departing from a default, changing your mind, and changing by a lot are different economic frictions and are
    modeled as three separate charges. They are *not* three parameterizations of one construct, and nothing here
    assumes they stand for the same thing:
        departure  c_dep * 1[S_t != S_D,t]        an overhead for not being on the default (for example the cost of
                                                  running a process that is not the standard one)
        switching  c_sw  * 1[S_t != S_{t-1}]      a fixed cost of changing the action at all (a setup or changeover)
        magnitude  c_mag * |S_t - S_{t-1}|        a cost proportional to how far the action moved
    The distinction matters: an agent that holds a constant non-default order pays the departure overhead every
    period and nothing for switching or magnitude, while an agent that tracks a moving default pays no departure
    overhead but pays for every change. Before 9 October 2026 only the departure overhead existed and the same
    indicator also *defined* the reported adaptation rate, so "how often the policy departed from the default" and
    "how often the policy actually changed its order" could not be told apart. They are now reported separately as
    `departure_rate` and `adjustment_rate`. `Config.cost_model = "legacy_departure_only"` restores the old
    calculation exactly (the two new charges are ignored, whatever they are set to).

Conventions fixed once, so the components are comparable
    tolerance     two orders count as different when they differ by more than ACTION_TOL (1e-9); the same tolerance
                  defines departure, switching and the "missed opportunity" accounting
    first period  period 0 has no previous action, so it is charged no switching and no magnitude cost; it is
                  charged the departure overhead if its order differs from that period's default
    burn-in       costs are charged in every period, and the reported means cover the recorded periods t >= burn_in.
                  The switching and magnitude charges at t = burn_in compare with the order in period burn_in - 1,
                  which is a real decision, so no recorded period is charged against an undefined predecessor.
    objective     tuning scores exactly the evaluation objective, `net_payoff` = gross payoff minus all charges that
                  the configuration switches on, so a policy is never tuned against a different economic objective
                  from the one it is judged by.

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
    primary     net payoff per period = gross payoff - departure - switching - magnitude charges, where the gross
                payoff is -(H_OVER (S - d)+ + P_SHORT (d - S)+). Reported alongside it: gross_payoff, the three
                charges separately (cost_departure, cost_switching, cost_magnitude, cost_total), the share of
                periods departing from the default (departure_rate), the share actually changing the order
                (adjustment_rate) and the mean |S_t - S_{t-1}| (adjustment_magnitude)
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
* **NC3 · Stable environment with abundant feedback.** No regime changes, every demand observed, little noise, 3,000 periods. Expected: R < 1 and the confidence-sensitive gate is at least as good as the better fixed rule. Estimand: the gate's mean paired advantage over that rule, in net payoff per period. Noninferiority test at a prespecified margin of 0.25 per period: passes if R < 1 and the *lower* limit of the 95% bootstrap CI of the advantage exceeds −0.25, so that the gate being worse by more than the margin is ruled out. Until 8 October 2026 the rule compared the *upper* limit with −0.25, which only fails when the gate is confidently worse and so established nothing.

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
   "switch_cost",
   0.0
  ],
  [
   "magnitude_cost",
   0.0
  ],
  [
   "cost_model",
   "components"
  ],
  [
   "perfect",
   false
  ]
 ],
 "burn_in": 100,
 "code": "ec6bb2821627c766afaaa005b324fda5c527354096bc99217be2f1df4749883e",
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
   "No regime changes, every demand observed, little noise, 3,000 periods. Expected: R < 1 and the confidence-sensitive gate is at least as good as the better fixed rule. Estimand: the gate's mean paired advantage over that rule, in net payoff per period. Noninferiority test at a prespecified margin of 0.25 per period: passes if R < 1 and the *lower* limit of the 95% bootstrap CI of the advantage exceeds \u22120.25, so that the gate being worse by more than the margin is ruled out. Until 8 October 2026 the rule compared the *upper* limit with \u22120.25, which only fails when the gate is confidently worse and so established nothing."
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
