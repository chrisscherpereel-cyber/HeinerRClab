import pandas as pd
import streamlit as st

from heiner_abm import registered
from heiner_abm.arena import (ALL_DESIGNS, SELECT_TEXT, TARGET_TEXT, THEORY_DESIGNS, THEORY_NAMES, VARIANTS, Composite,
                              Spec)
from heiner_abm.calibration import FORECAST_RULES, QUANTITY_RULES
from heiner_abm.empirical import NEWSVENDOR_RULES
from heiner_abm.experiment import RIVALS, SHADOWS
from heiner_abm.literature import REFERENCES
from heiner_abm.params import SELECTION_HELP
from heiner_abm.rulechoice import RESTRICTED, RULE_LABELS, RULES

st.title("Agents as implemented")
st.caption("Every decision-making agent in the laboratory, exactly as the code implements it: what it observes, how it "
           "decides, its parameters and where it is used. Parameter tables and rule descriptions are read from the "
           "code itself, so they cannot drift from what is simulated.")

# ------------------------------------------------------------------------------------------------ overview
st.header("1 · Where the agents appear", divider="gray")
st.dataframe(pd.DataFrame([
    ("Market-lab firms", "Market lab, Hypothesis tests, Risk vs Knightian uncertainty, Endogenous flexibility, "
                         "Competing theories, Experiment designer, Heiner special tests",
     "Firms that differ in flexibility φ and share a production rule (Bertrand or Cournot) and a selection rule."),
    ("Tournament designs", "Agent tournament, Mechanisms, Rule choice, Field patterns, Signature tests, Play the market",
     "Seventeen designs, two per theory plus rule B, each implementing one theory's decision rule."),
    ("Mechanism variants", "Heiner: mechanisms", "The reliability-condition agent judging with the true model, and "
                                                "an oracle that knows its rule's true reliability."),
    ("Rule-choosing firms", "Rule choice (emergence)", "Firms that switch between six tournament designs according to "
                                                       "their recent performance."),
    ("Task agents", "Generalization", "A slow default and a fast flexible estimate in inventory, learning and investment "
                                      "tasks, with five selection layers."),
    ("Tracking rules", "Heiner vs optimal filtering", "A single firm tracking a moving target: Kalman filter, filter with "
                                                       "offset, partial adjustment and restricted filters."),
    ("Rules fitted to people", "Calibration, Empirical validation, Experiment analysis",
     "Forecasting, quantity and newsvendor rules fitted to human choices, and the designs used as shadows."),
], columns=["Agents", "Pages", "What they are"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ common cycle
st.header("2 · The shared market and decision cycle", divider="gray")
st.markdown(
    "All firm agents act in the same cobweb oligopoly: a homogeneous product, linear demand with a price floor, "
    "and a raw-material cost that follows a bounded random walk. Each period:\n\n"
    "1. **Observe.** Every firm sees last period's market price P, total market quantity Q, its own output q and "
    "profit, and an estimate ĉ of the coming unit cost of the same quality for every firm:")
st.latex(r"\hat c_t = c_{t-1} + \kappa\,(c_t - c_{t-1}) + \sigma\,\varepsilon_t,\qquad \varepsilon_t\sim N(0,1)")
st.markdown(
    "   κ is competence (foresight; 0 means the firm sees only last period's cost) and σ is perception error.\n"
    "2. **Decide.** The agent's rule turns this information into an output (rounded to whole units, at least the "
    "minimum production). Model-based agents use their *believed* demand curve, which after an unannounced shift "
    "stays at the old regime for L periods (the model-updating lag).\n"
    "3. **Clear.** The market clears on the true demand curve and the new cost is realized:")
st.latex(r"P_t = \max\!\big(P_{\min},\; P^{\max}_t - s_t\,Q_t\big),\qquad \pi_{i,t} = (P_t - c_t)\,q_{i,t} - F_i")
st.markdown(
    "4. **Learn.** Agents that learn update their state (filters, values, propensities, aspirations or reliability "
    "estimates). Learning agents never see the future: anything judged *ex post* is judged only after the periods it "
    "refers to have been played.\n\n"
    "Every agent receives the same information; agents differ only in how they turn it into output. Random draws "
    "for costs, perception errors, demand shifts and the agents' own exploration come from separate streams, so the "
    "same seed reproduces the same market for every agent (common random numbers).")

# ------------------------------------------------------------------------------------------------ market-lab firms
st.header("3 · Market-lab firms", divider="gray")
st.markdown(
    "The firms on the Market lab, Hypothesis tests and related pages share one production rule and one selection "
    "rule, chosen in the sidebar, and differ in flexibility φᵢ = intercept + slope·i (firm 1 is the most rigid).\n\n"
    "**Production rule: the recommended output q\\*.**")
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**Cournot (model-based)**: move a share φ of the way toward the best reply on the believed demand "
                "curve, given rivals' last output R = Q − q:")
    st.latex(r"q^* = \varphi\,\frac{\hat P^{\max} - \hat s\,R - \hat c}{2\hat s} + (1-\varphi)\,q")
with c2.container(border=True):
    st.markdown("**Bertrand (model-free)**: adjust output to the observed margin relative to a desired margin m\\* "
                "(minus F/q if fixed costs are included in the margin):")
    st.latex(r"q^* = q + \varphi\,\big(P - \hat c - m^*\big)")
st.markdown("**Selection rule: deviate from rule B (keep q) and adopt q\\*?**")
st.dataframe(pd.DataFrame([(k, t) for k, t in SELECTION_HELP.items()], columns=["Rule", "Behavior"]),
             hide_index=True, width="stretch")
st.markdown(
    "The **Adaptive** rule groups recommended changes |q\\* − q| into five size bins (edges 5, 15, 30 and 60 units). "
    "For each bin it keeps a learned gain E_b. A change of that size recommended at period d is judged at the end of "
    "period d + W − 1, once every period it covers has been observed:")
st.latex(r"E_b \leftarrow \lambda\,E_b + (1-\lambda)\,g_d,\qquad g_d = \sum_{s=d}^{d+W-1}\big[(\tilde P_s(x_1)-c_s)\,x_1"
         r" - (\tilde P_s(x_0)-c_s)\,x_0\big],\quad \tilde P_s(x) = \max\{P_{min},\ \tilde P_{max,s} - \tilde s_s(R_s + x)\}")
st.markdown(
    "where x₁ is the recommended output, x₀ the old output (rule B), R_s the rivals' actual output, c_s the realized "
    "cost and P̃ the firm's *believed* demand curve. Every term is known to the firm by the end of period d + W − 1, so "
    "it learns only from information available to it (the same judgement as the tournament's reliability-condition "
    "agents). The firm deviates in a bin only if E_b ≥ 0. Nothing tells it how often to deviate: restraint, if it "
    "appears, is learned.\n\n"
    "*Oracle treatment (researcher-only).* With *oracle* feedback (sidebar), g is instead the researcher's "
    "counterfactual over the measurement horizon H, computed by forking the market with the true demand curve and "
    "rivals' simulated reactions. It is held in a pending queue and released at the end of period d + H − 1, once "
    "every period it covers has occurred. No firm could compute it; results with it are oracle benchmarks.\n\n"
    "**Cost of flexibility.** Each firm pays Fᵢ = a·φᵢ + b per period. **Endogenous flexibility** (Evolution page): "
    "every 50 periods each firm copies the flexibility of the most profitable firm with probability 0.5 and adds a "
    "small mutation (s.d. 0.03), within [0, 1].")

# ------------------------------------------------------------------------------------------------ tournament designs
st.header("4 · Tournament designs: one theory, two decision rules", divider="gray")
st.markdown(
    "Each theory is implemented as two agent designs; each design is tuned with the same budget on training "
    "environments, and the better of the two enters the tournament. Many designs combine a **target** (where a "
    "flexible firm would move) with a **selection rule** (when it moves), so selection rules can be compared on "
    "the same target.")
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**Targets**")
    st.markdown("*Model-based*: " + TARGET_TEXT["model"] + ".")
    st.latex(r"\tilde c \leftarrow \tilde c + a_c(\hat c - \tilde c),\;\; \tilde R \leftarrow \tilde R + a_R(R - "
             r"\tilde R),\;\; x^* = \frac{\hat P^{\max} - \hat s\tilde R - \tilde c}{2\hat s}")
    st.markdown("*Price-based*: " + TARGET_TEXT["price"] + ".")
    st.latex(r"p^e \leftarrow p^e + \lambda(P - p^e),\qquad x^* = \frac{p^e + \hat s\,q - \hat c}{2\hat s}")
    st.markdown("Either way the candidate output is q̃ = round(q + φ·(x\\* − q)).")
with c2.container(border=True):
    st.markdown("**Selection rules**")
    st.markdown(
        f"* *Always*: {SELECT_TEXT['always']}.\n"
        f"* *Inaction band*: {SELECT_TEXT['band']}: v ← 0.9·v + 0.1·(Δx\\*)², move if |q̃ − q| > k·√v.\n"
        f"* *Reliability condition*: {SELECT_TEXT['rc']}. Changes are binned by size (edges θ·0.5, θ, 2θ, 4θ); "
        "each bin's gain E_b ← m·E_b + (1 − m)·g is learned from the profit of holding q̃ rather than q over the "
        "next h periods (rivals on their actual path, the agent's believed demand, realized costs); the agent "
        "deviates only where E_b ≥ 0.\n"
        f"* *Aspiration*: {SELECT_TEXT['aspiration']}: A ← A + α·(π − A), move if last profit < A.")

DETAIL = {
    "opt_nash": r"Target $x^* = (\hat P^{\max} - \tilde c)/(\hat s\,(N+1))$, the symmetric Cournot–Nash output at the "
                r"filtered cost; $q \leftarrow q + \varphi\,(x^* - q)$.",
    "heur_wsls": r"Keeps a direction $d \in \{-1, +1\}$; reverses it when profit fell since last period; "
                 r"$q \leftarrow q + d\cdot\text{step}$.",
    "heur_markup": r"$q \leftarrow q + \varphi\,(P - \hat c - m)$: expand when the observed margin exceeds the target "
                   r"margin $m$, contract otherwise.",
    "rl_softmax": r"Moves $a \in \{-2,-1,0,+1,+2\}\times\text{step}$. After each move, its value is updated with the "
                  r"profit change $r$ that followed: $V_a \leftarrow V_a + \eta\,(r - V_a)$. Next move drawn with "
                  r"probability $\propto \exp\!\big(V_a / (\text{temp}\cdot s)\big)$, $s$ = running size of profit "
                  r"changes.",
    "rl_erevroth": r"Same five moves. Propensities decay by the forgetting rate each period and the chosen move's "
                   r"propensity grows by $\text{gain}\cdot\max(r, 0)/s$; moves are drawn in proportion to "
                   r"propensities.",
    "imit_best": r"With probability $p$: copy last period's output of the most profitable firm; otherwise keep $q$. "
                 r"Add experimentation noise $\text{noise}\cdot z$.",
    "imit_avg": r"With probability $p$: move to the average output of the other firms; otherwise keep $q$. Add "
                r"experimentation noise.",
    "ruleb": r"$q_t = q_{t-1}$ always (Heiner's rule B: the rigid benchmark).",
}


def _params(cls) -> pd.DataFrame:
    rows = []
    tuned = registered.TUNED_PARAMS.get(cls.key, {}) if hasattr(registered, "TUNED_PARAMS") else {}
    for n, sp in cls.SPACE.items():
        if not isinstance(sp, Spec):
            continue
        rows.append({"Parameter": n, "Meaning": sp.help, "Default": sp.default, "Range": f"{sp.lo:g}–{sp.hi:g}",
                     "Search scale": sp.scale, "Registered tuned value": tuned.get(n)})
    return pd.DataFrame(rows)


def _design(key: str):
    cls = ALL_DESIGNS[key]
    entered = registered.TUNED_DESIGN if hasattr(registered, "TUNED_DESIGN") else {}
    tag = " · **entered the registered tournament**" if key in entered.values() else ""
    st.markdown(f"**{cls.name}** (`{key}`){tag}")
    st.markdown(cls.rule)
    if issubclass(cls, Composite):
        st.caption(f"Target: {cls.TARGET}-based · selection: {cls.SELECT}"
                   + (f" · fixed: {cls.FIXED}" if cls.FIXED else ""))
    elif key in DETAIL:
        st.markdown(DETAIL[key])
    p = _params(cls)
    if len(p):
        st.dataframe(p, hide_index=True, width="stretch")
    st.caption("Sources: " + "; ".join(REFERENCES[s].cite for s in cls.sources if s in REFERENCES))


for theory, keys in THEORY_DESIGNS.items():
    with st.expander(f"{THEORY_NAMES[theory]} · {', '.join(ALL_DESIGNS[k].name for k in keys)}"):
        for k in keys:
            _design(k)

with st.expander("Mechanism variants of the reliability-condition agent"):
    st.markdown("Used only on the *Heiner: mechanisms* page, not in the tournament. **Judged with the true model**: "
                "the same learner, but each past decision is evaluated with the true demand curve instead of the "
                "agent's believed one (removes model bias). **Oracle**: does not learn; each size bin's expected gain "
                "E_b is fixed in advance at the true value for its environment, estimated from a long independent run "
                "(it knows its rule's reliability, not the future).")
    for k in VARIANTS:
        st.markdown(f"* **{ALL_DESIGNS[k].name}** (`{k}`)")

# ------------------------------------------------------------------------------------------------ other populations
st.header("5 · Other agent populations", divider="gray")
with st.expander("Rule-choosing firms (Rule choice page)"):
    st.markdown(
        "Every firm carries six of the tournament designs at once: " +
        ", ".join(f"{RULE_LABELS[r]}{' (restricted)' if r in RESTRICTED else ''}" for r in RULES) + ". "
        "All six observe the firm's actual output and propose an output each period, so switching loses no state; "
        "the firm executes the proposal of its current rule. Each period it revises its rule with a small "
        "probability, choosing rule k with logit probability")
    st.latex(r"P(k) = \frac{\exp(\beta\,U_k / s)}{\sum_j \exp(\beta\,U_j / s)}")
    st.markdown("where U_k is the exponentially weighted average profit of the firms currently using rule k, s a "
                "running profit scale and β the intensity of choice (Brock & Hommes, 1997).")
with st.expander("Task agents (Generalization page)"):
    st.markdown(
        "In the inventory, learning and investment tasks the **default** is a slow, long-memory estimate (smoothing "
        "gain 0.01) and the **flexible** alternative a fast re-estimate (gain g, varied across environments). Five "
        "selection layers decide each period whether to deviate from the default to the flexible action: *always*; "
        "an *inaction band* (deviate when the two differ by more than b noise units; b tuned on training "
        "environments); a *learned reliability condition* (per signal-strength bin, a running mean of the observed "
        "gain of deviating; deviate where positive); an *oracle* reliability condition (the true mean gain per bin "
        "from a long independent run); and *rule B* (never deviate).")
with st.expander("Tracking rules (Heiner vs optimal filtering page)"):
    st.markdown(
        "A single firm tracks a random-walk target observed with noise. **Kalman filter**: moves to the filtered "
        "estimate every period (gain k = P/(P + r)). **Filter + offset**: the same plus the loss-minimizing offset "
        "for lopsided stakes (the Bayes-optimal rule). **Partial adjustment** with one speed, or separate speeds up "
        "and down. **Restricted filter**: moves to the filtered estimate only when the move exceeds a threshold, "
        "with separate thresholds up and down (reliability-based restriction).")
with st.expander("Rivals and shadows in the human experiment (Play the market)"):
    st.markdown("Participants face three tuned rivals: " + ", ".join(ALL_DESIGNS[k].name for k in RIVALS) + ". "
                "Alongside each choice, these designs record in shadow mode what they would have chosen in the "
                "participant's place (used to classify participants): "
                + ", ".join(ALL_DESIGNS[k].name for k in SHADOWS) + ".")
with st.expander("Rules fitted to human choices (Calibration and Empirical validation)"):
    for title, rules in (("Forecasting rules", FORECAST_RULES), ("Quantity (Cournot) rules", QUANTITY_RULES),
                         ("Newsvendor rules", NEWSVENDOR_RULES)):
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame([dict(Rule=r.name, Theory=r.theory,
                                        Parameters=", ".join(f"{k} ({len(v)} values)" for k, v in r.grid.items())
                                        or "none") for r in rules]), hide_index=True, width="stretch")
    st.caption("Each rule's parameters are fitted per participant on the first half of the periods (grid search) and "
               "the rule is scored on the second half.")
