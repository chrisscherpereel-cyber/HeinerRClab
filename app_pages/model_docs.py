import pandas as pd
import streamlit as st

from heiner_abm.literature import bibliography
from ui.common import DEFAULTS, PRESET_INFO, PRESETS, PREREG_TEXT, replication_notice

MODEL_REFS = ["heiner1983", "heiner1986", "heiner1988", "heiner1988b", "heiner1989", "ezekiel1938", "nerlove1958",
              "theocharis1960", "carlson1967", "carlson1968", "knight1921", "green1966", "scarf1960", "lindblom1959",
              "erev1998", "hanley1982", "efron1993", "tesfatsion2006", "nosek2018"]

st.title("Model & methods")

st.header("Agents and schedule", divider="gray")
st.markdown(
    r"""
The model is an agent-based cobweb oligopoly. One homogeneous-product, market-clearing market is populated by
$n$ firm agents. Each period $t = 1,\dots,T$ runs in this order:

1. **Perceive.** Firm $i$ sees last period's price $P_{t-1}$, market quantity $Q_{t-1}$ and its own production $q_{i,t-1}$.
   It forms a cost estimate $\hat c_{i,t} = c_{t-1} + \kappa_i (c_t - c_{t-1}) + \sigma_i \varepsilon_{i,t}$.
   Baseline firms have $\kappa = \sigma = 0$: they decide on last period's cost.
2. **Recommend.** The firm's production rule proposes $q^*_{i,t}$ (rounded to an integer, at least $q_{min}$).
3. **Select.** The firm's selection rule decides whether to deviate from the default rule
   **B: $q_{i,t} = q_{i,t-1}$** and adopt $q^*$.
4. **Clear.** $Q_t = \sum_i q_{i,t}$, $P_t = \max\{P_{min},\ P_{max} - sQ_t\}$ with $s = (P_{max}-P_{min})/Q_{range}$.
5. **Shock.** The new raw-material cost is realized:
   $c_t = \text{reflect}(c_{t-1} + \Delta\, U_t)$ with $U_t\sim\mathcal U(-1,1)$ and reflecting bounds $[P_{min}, c_{max}]$.
6. **Book.** Profit $\pi_{i,t} = (P_t - c_t)\,q_{i,t} - F_i$ with $F_i = a\varphi_i + b$.
7. **Evaluate (researcher).** The counterfactual payoff of the other choice is computed (below) to measure
   $\pi, r, w, G, D$. This uses periods $t \ldots t+H-1$, the true demand curve and rivals' true rules, so it is
   never shown to ordinary agents.
8. **Queue feedback.** Each decision with an opportunity ($q^* \ne q$) enters a pending-feedback queue with its
   maturity: $t + W - 1$ for observable feedback (its gain accumulates period by period from what the firm observes:
   rivals' actual output, realized costs, prices on its believed demand curve), or $t + H - 1$ for the
   researcher-only *oracle* treatment (the counterfactual of step 7).
9. **Release.** Every item whose last included period is $t$ is released, in order of decision time, and updates the
   firm's learned table. Released feedback is first used in the decisions of period $t+1$. Items that would mature
   after the last period are never released (no partial feedback).
10. **Evolve** (optional): firms revise $\varphi$ by imitating the most profitable rival (realized profits).

Steps 1–2 use only information from period $t-1$ and earlier, except that a firm with cost foresight $\kappa > 0$
anticipates that share of the coming cost change (a competence parameter; $\kappa = 0$ in the baseline). The same
schedule governs the vectorized engine, the tournament agents and the interactive market
(`heiner_abm/agents.py`, `DECISION_SCHEDULE`; `tests/test_information.py`).
""")

c1, c2 = st.columns(2)
with c1:
    st.markdown(r"""
**Cournot rule** (partial adjustment toward the best reply, assuming rivals keep last period's output)

$$q^{BR}_{i} = \frac{P_{max} - s\,(Q_{t-1} - q_{i,t-1}) - \hat c_{i,t}}{2s},\qquad q^*_{i,t} = \varphi_i\, q^{BR}_i + (1-\varphi_i)\, q_{i,t-1}$$

$\varphi_i\in[0,1]$: 0 = inflexible, 1 = full best reply.
""")
with c2:
    st.markdown(r"""
**Bertrand rule** (margin feedback)

$$q^*_{i,t} = q_{i,t-1} + \varphi_i\big[(P_{t-1} - \hat c_{i,t} - \mathbb 1_{incl}F_i/q_{i,t-1}) - m^*_i\big]$$

$m^*$ = desired margin (competition intensity). $\varphi_i\ge 0$ = sensitivity.
""")

st.markdown(
    """
| Selection rule | Deviates from B when… | Grounded in |
|---|---|---|
| **Always** | always (maximally flexible) | frictionless optimization |
| **Never** | never (rule B only, rigid) | rule-governed behavior (Heiner 1983) |
| **Small** (SR1) | \\|q* − q\\| < θ ("change is risky") | incrementalism (Lindblom 1959) |
| **Large** (SR2) | \\|q* − q\\| > θ ("big imbalances send clear signals") | signal detection, (S, s) inaction bands (Green & Swets 1966; Scarf 1960) |
| **Adaptive** | its learned average gain for this size of change ≥ 0 (exponential memory λ, 5 size bins, judged from observed outcomes over W periods) | reinforcement learning (Erev & Roth 1998) |
""")

st.header("Measuring Heiner's quantities", divider="gray")
st.markdown(
    r"""
An **opportunity** is a period in which $q^* \ne q_{t-1}$. For each opportunity, the payoff of adopting $q^*$ is compared
with the payoff of following B (keeping $q_{t-1}$) under the realized cost:

* **H = 1 (Heiner's one-shot comparison).** One-period profit, rivals' period-$t$ choices held fixed:
  $\text{gain} = \pi_i(q^*) - \pi_i(q_{t-1})$.
* **H > 1 (extension).** The market is **forked** at $t$ into two branches that differ only in firm $i$'s choice. Both
  branches run $H$ periods with every agent following its rules and identical random shocks. The gain is the
  difference in firm $i$'s cumulative profit. By default the evaluated firm returns to **rule B** after the decision,
  which is Heiner's "deviate at this instance, otherwise follow B". Alternatively it keeps its own rules, which gives
  a timing comparison.

Then per firm (after burn-in):

$$\pi = \frac{\#\text{exceptions}}{\#\text{opportunities}},\quad r = P(\text{dev}\mid\text{exception}),\quad
w = P(\text{dev}\mid\text{no exception}),$$
$$G = E[\text{gain}\mid\text{dev, exception}],\quad D = E[-\text{gain}\mid\text{dev, no exception}],\qquad
\text{RC: } \frac{r}{w} > \frac{D}{G}\frac{1-\pi}{\pi}.$$

If a firm never deviated in a state, G or D falls back to the average *potential* gain or loss in that state.
The **measured CD-gap** is $\sqrt{E[(\hat c_{i,t} - c_t)^2]}$, the RMSE of the firm's cost perception.

**Identity check.** The sum of instance gains over deviations equals $\sum$ correct-deviation gains minus
$\sum$ wrong-deviation losses. The RC is exactly the condition that this sum is positive, and the test suite
verifies the identity.
""")

st.header("Dynamic, out-of-sample and structural extensions", divider="gray")
st.markdown(
    r"""
**Three measures of a decision's value.** For each opportunity, the gain of following the rule instead of B is
computed three ways (all with discount $\gamma$ over horizon $H$; all equal when $H = 1$):

| Measure | Firm after the decision | Rivals | Captures |
|---|---|---|---|
| *static* | (one period only) | choices fixed | Heiner's one-shot comparison |
| *persist* | keeps the chosen level (rule B) | follow their **actual** path, no reaction | + persistence of the decision |
| *full* | rule B (or its own rules) | **react** in a forked market | + strategic feedback |

Decomposition: full = static + (persist − static) + (full − persist). The *persist* measure is computed after the
run from recorded paths; the *full* measure forks the market 2N times per period.

**Out-of-sample windows.** Recorded periods are split at a chosen share (default 50%) into an *estimation* and an
*evaluation* window. π, r, w, G, D are computed separately in each window, so the RC estimated in the first
window can be tested against flexible-vs-rigid profits in the second. Predictions are scored by AUC (the
Mann–Whitney probability that a random winner scores higher than a random loser), with 95% CIs from a bootstrap
that resamples whole environments.

**Heiner (1989) error-to-signal ratio.** For each firm and period, the perceived target is the Cournot best reply
computed with the firm's cost estimate and believed demand, $\hat x_t$. The true target $x^*_t$ is the ex-post best
reply to rivals' actual output, the realized cost and the true demand. $\xi_t = \hat x_t - x^*_t$ and
$\Delta^*_t = x^*_t - q_{t-1}$. The reported $K = \mathrm{RMS}(\xi)/\mathrm{RMS}(\Delta^*)$ is a robust version of
the theorem's bound on $|\xi/\Delta^*|$. The bound is $\beta_0 = 1/((1+K)(1-f'))$ with $f' = -(n-1)/2$.

**Structural (Knightian) uncertainty.** With hazard λ per period a new demand regime is drawn around the baseline:
$P_{max}' = P_{max} + s_1 z_1$ and $s' = s\,e^{s_2 z_2}$ (clipped). Markets clear on the true curve. Cournot firms
compute best replies with the regime of $L$ periods earlier ($L = -1$: never updated). Bertrand firms use only
observed prices. Regime draws come from a separate random stream, so switching them off reproduces the baseline
exactly.

**Signal-detection ROC.** For the decision signal $|q^* - q|$ and preferred-exception labels, sweeping a threshold θ
gives $(w(\theta), r(\theta))$. The signal's AUC is the firm's discriminability, and the value-maximizing θ is the
optimal SR2 threshold.
""")

st.header("Experimental design & statistics", divider="gray")
st.markdown(
    """
* **Replications & common random numbers.** Replication *r* uses seed = base seed + *r* in every condition, so
  conditions are compared on identical cost shocks. Randomness comes from independent streams for the cost path,
  perception noise and evolution, so changing the number of firms does not change the cost path.
* **Burn-in.** The first periods are discarded (baseline: 25).
* **Payoff to flexibility.** Within each market, the OLS slope of average profit on φ. Conditions report the mean
  slope with a t-based 95% CI across replications, the Spearman rank correlation, and how often the most or least
  flexible firm earns the most.
* **Switch point.** Linear interpolation where the mean slope changes sign.
* **Rigid twin.** For RC validation, each firm's market is re-run with that firm set to *Never* (same shocks). The
  dynamic advantage of flexibility is its profit difference.
* **Regimes.** Lag-1 serial correlation of price versus cost: price > cost means sluggish, ≈ means tracking,
  < means oscillating.
""")

st.header("Baseline calibration", divider="gray")
st.dataframe(pd.DataFrame([
    ("Demand", "P = max(P_min, P_max − s·Q), s = (P_max − P_min)/Q_range", "P_max 100, P_min 10, Q_range 1500"),
    ("Raw-material cost", "Reflecting random walk c_t = c_{t−1} + Δ·U(−1, 1) on [P_min, c_max]", "c₀ 45, c_max 80, Δ 10"),
    ("Timing", "Firms decide on last period's price, quantity and cost; the new cost is drawn after clearing", "–"),
    ("Firms", "Flexibility φᵢ = slope·i + intercept, ordered from most rigid to most flexible", "4 firms, slope 0.25"),
    ("Production", "Rounded to whole units, at least the minimum production", "q₀ 200, q_min 15"),
    ("Bertrand", "Desired margin m*", "m* 5"),
    ("Selection rules", "Threshold θ for Small and Large", "θ 25"),
    ("Flexibility cost", "Fᵢ = a·φᵢ + b", "a = b = 0"),
    ("Measurement", "Burn-in, horizon H, discount γ, estimation window", "25 periods, H 20, γ 1, 50%"),
], columns=["Element", "Specification", "Baseline"]), hide_index=True, width="stretch")

st.header("Preset scenarios", divider="gray")
st.markdown("The sidebar's presets are ready-made variations of the baseline. Pick one, press **Apply preset**, then "
            "run any page: each preset changes only the settings listed, so its effect can be read against the "
            "baseline. The typical results come from six replications of 1,000 periods; *slope* is the change in "
            "average profit per period from one firm to the next more flexible firm (positive: flexibility pays).")
rows = []
for name, over in PRESETS.items():
    info = PRESET_INFO.get(name, {})
    changes = ", ".join(f"{k} = {v}" for k, v in over.items() if DEFAULTS.get(k) != v) or "none (the baseline)"
    rows.append({"Preset": name, "Settings changed": changes, "Setup": info.get("setup", ""),
                 "Why run it": info.get("why", ""), "Used for": info.get("hypotheses", ""),
                 "Typical result": info.get("typical", "")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
replication_notice("presets", "preset_adaptive")

st.header("Established theory, reduced form and proposed extensions", divider="gray")
st.markdown("Each part of the laboratory is in one of three classes. **Established theory**: a published result, used or "
            "reproduced as stated. **Reduced-form implementation**: the laboratory's stylized stand-in for a market "
            "feature or a theory's mechanism, a modeling choice rather than a claim of the cited work. **Proposed "
            "extension**: a construct the laboratory adds; its results test that construct, not the original theory.")
from heiner_abm.claim_status import COMPONENTS, STATUS_LABELS  # noqa: E402
from heiner_abm.literature import REFERENCES as _REFS  # noqa: E402
st.dataframe(pd.DataFrame([{
    "Status": STATUS_LABELS[c.status], "Component": c.name, "What it is": c.what,
    "Sources": "; ".join(_REFS[k].cite for k in c.sources) or "the laboratory's own", "Where": c.where}
    for c in COMPONENTS]), hide_index=True, width="stretch")

st.header("Status of reported findings without a frozen plan", divider="gray")
st.markdown("Findings from a frozen plan are tied to its hash. The findings below were produced without one, so each is "
            "tied to a fingerprint of the source files that produce it. When those files change, the finding is "
            "marked *requires replication* here, on the pages that report it and in the README; its numbers are left "
            "as originally reported until the study is rerun.")
from heiner_abm import registered as _reg  # noqa: E402
st.dataframe(pd.DataFrame([{
    "Finding": title, "Source files": ", ".join(f.replace("#tuned", "registered tuned parameters") for f in files),
    "Status": "requires replication" if _reg.replication_note(k) else "current (code unchanged since reported)",
    "Note": _reg.replication_note(k) or ""} for k, (title, files) in _reg.FINDING_SOURCES.items()]),
    hide_index=True, width="stretch")

st.header("Pre-registered plans", divider="gray")
st.markdown(PREREG_TEXT)

st.header("Verification", divider="gray")
st.markdown(
    """
The model exists twice: a readable object-oriented agent implementation (`heiner_abm/agents.py`: `Firm`, `Market`,
`Industry`) and a vectorized batch engine (`heiner_abm/engine.py`) for Monte Carlo work. The test suite
(`pytest tests`) checks that they produce **identical** trajectories and counterfactuals for Bertrand, Cournot and
mixed markets, with and without forks, evolution, discounting and demand-regime shifts. The persistence
measure is checked against a brute-force loop, and the estimation and evaluation windows are checked to partition
the recorded periods. It also checks a reflecting-boundary example
(75 + 15 with bound 80 → 70), the cost bounds, the rigid 'Never' rule, and the RC accounting identity.
""")

st.header("References", divider="gray")
st.markdown("\n".join(f"* {r.apa}" for r in bibliography(MODEL_REFS)))
st.caption("The research behind every hypothesis, and the complete bibliography, are on the *Research & "
           "contribution* page.")
