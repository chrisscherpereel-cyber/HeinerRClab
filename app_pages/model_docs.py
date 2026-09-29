import pandas as pd
import streamlit as st

st.title("Model & methods")

st.header("Agents and schedule", divider="gray")
st.markdown(
    r"""
The model is an agent-based cobweb oligopoly. One homogeneous-product, market-clearing market is populated by
$n$ firm agents. Each period $t = 1,\dots,T$ runs in this order:

1. **Perceive.** Firm $i$ sees last period's price $P_{t-1}$, market quantity $Q_{t-1}$ and its own production $q_{i,t-1}$.
   It forms a cost estimate $\hat c_{i,t} = c_{t-1} + \kappa_i (c_t - c_{t-1}) + \sigma_i \varepsilon_{i,t}$.
   The paper's firms have $\kappa = \sigma = 0$: they decide on last period's cost.
2. **Recommend.** The firm's production rule proposes $q^*_{i,t}$ (rounded to an integer, at least $q_{min}$).
3. **Select.** The firm's selection rule decides whether to deviate from the default rule
   **B: $q_{i,t} = q_{i,t-1}$** and adopt $q^*$.
4. **Clear.** $Q_t = \sum_i q_{i,t}$, $P_t = \max\{P_{min},\ P_{max} - sQ_t\}$ with $s = (P_{max}-P_{min})/Q_{range}$.
5. **Shock.** The new raw-material cost is realised:
   $c_t = \text{reflect}(c_{t-1} + \Delta\, U_t)$ with $U_t\sim\mathcal U(-1,1)$ and reflecting bounds $[P_{min}, c_{max}]$.
6. **Book.** Profit $\pi_{i,t} = (P_t - c_t)\,q_{i,t} - F_i$ with $F_i = a\varphi_i + b$.
7. **Evaluate & learn.** The counterfactual payoff of the other choice is computed (below). Adaptive agents update
   their memories, and firms optionally revise $\varphi$ (evolution).
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
| Selection rule | Deviates from B when… | Source |
|---|---|---|
| **Always** | always (maximally flexible) | VBA |
| **Never** | never (rule B only, rigid) | VBA |
| **Small** (SR1) | \\|q* − q\\| < θ ("change is risky") | paper & VBA |
| **Large** (SR2) | \\|q* − q\\| > θ ("big imbalances send clear signals") | paper & VBA |
| **Adaptive** | its learned average gain for this size of change ≥ 0 (exponential memory λ, 5 size bins) | new |
""")

st.header("Measuring Heiner's quantities", divider="gray")
st.markdown(
    r"""
An **opportunity** is a period in which $q^* \ne q_{t-1}$. For each opportunity, the payoff of adopting $q^*$ is compared
with the payoff of following B (keeping $q_{t-1}$) under the realised cost:

* **H = 1 (the paper's proposal).** One-period profit, rivals' period-$t$ choices held fixed:
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

st.header("Experimental design & statistics", divider="gray")
st.markdown(
    """
* **Replications & common random numbers.** Replication *r* uses seed = base seed + *r* in every condition, so
  conditions are compared on identical cost shocks. Randomness comes from independent streams for the cost path,
  perception noise and evolution, so changing the number of firms does not change the cost path.
* **Burn-in.** The first periods are discarded (paper: 25).
* **Payoff to flexibility.** Within each market, the OLS slope of average profit on φ. Conditions report the mean
  slope with a t-based 95% CI across replications, the Spearman rank correlation, and how often the most or least
  flexible firm earns the most. This follows the paper's plan to relate flexibility to profits by regression across
  20 replications.
* **Switch point.** Linear interpolation where the mean slope changes sign.
* **Rigid twin.** For RC validation, each firm's market is re-run with that firm set to *Never* (same shocks). The
  dynamic advantage of flexibility is its profit difference.
* **Regimes.** Lag-1 serial correlation of price versus cost: price > cost means sluggish, ≈ means tracking,
  < means oscillating.
""")

st.header("Parity with the 2006 VBA workbook", divider="gray")
st.dataframe(pd.DataFrame([
    ("Demand", "P = MaxPrice − Slope·Q, floored at MinPrice", "Identical"),
    ("Raw-material cost", "Adjustment = 2(Rnd − 0.5)·MaxChange, reflexive boundaries at MinPrice / MaxRawMatCost", "Identical"),
    ("Timing", "Firms produce on last round's price, quantity & cost; new cost drawn after clearing; profit uses new cost", "Identical"),
    ("Cournot", "(MaxPrice − Slope(MrktQ − q) − RMC)/(2·Slope), weighted by flexibility", "Identical"),
    ("Bertrand", "Flex·((Price − RMC) − DesiredProfit) + q", "Identical (+ optional F/q in margin as in the paper)"),
    ("Rounding / min production", "VBA Round (banker's), floor at MinProduction", "Identical (numpy rint = banker's)"),
    ("Accept rules", "Never / Always / Small / Large with threshold", "Identical + Adaptive"),
    ("Flexibility", "Flex = FlexSlope·i + FlexIntercept; FlexCost = FlexCostSlope·Flex + FlexCostIntercept", "Identical"),
    ("Profit", "(Price − RMC − FlexCost/q)·q", "Identical"),
    ("Firms / markets", "4 firms fixed; 'markets' = replications", "2–12 firms; any number of replications"),
    ("Reliability condition", "Not implemented (the paper lists it as future work)", "Implemented, with horizon H"),
    ("Competence (κ, σ)", "–", "New: makes the CD-gap an explicit, manipulable variable"),
    ("Endogenous flexibility", "–", "New: imitation + mutation of φ"),
], columns=["Element", "VBA (HeinerIOExp.xls)", "This model"]), hide_index=True, width="stretch")
st.caption("Defaults reproduce the VBA SetUpForm: MaxPrice 100, MinPrice 10, QuantityRange 1500, InitRawMatCost 45, "
           "MaxRawMatCost 80, MaxRawMatCostChange 10, InitProduction 200, MinProduction 15, FlexSlope 0.25, "
           "DesiredProfit 5, Threshold 25, record after period 25.")

st.header("Verification", divider="gray")
st.markdown(
    """
The model exists twice: a readable object-oriented agent implementation (`heiner_abm/agents.py`: `Firm`, `Market`,
`Industry`) and a vectorised batch engine (`heiner_abm/engine.py`) for Monte Carlo work. The test suite
(`pytest tests`) checks that they produce **identical** trajectories and counterfactuals for Bertrand, Cournot and
mixed markets, with and without forks and evolution. It also checks the reflecting-boundary example from the paper
(75 + 15 with bound 80 → 70), the cost bounds, the rigid 'Never' rule, and the RC accounting identity.
""")

st.header("References", divider="gray")
st.markdown(
    """
* Heiner, R. A. (1983). The origin of predictable behavior. *American Economic Review*, 73(4), 560–595.
* Scherpereel, C. M. & Summers, G. (2011). The influence of mistakes in Cournot and Bertrand competition: a simulation
  test of Heiner's reliability condition. *ABSEL*, Pensacola, FL.
* Carlson, J. A. (1968). An invariably stable cobweb model. *Review of Economic Studies*, 35(3), 360–362.
* Kampmann, C. & Sterman, J. (1998). Feedback complexity, bounded rationality, and market dynamics. MIT working paper.
""")
