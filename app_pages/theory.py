import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import CAT, style

st.title("Heiner's reliability condition")
st.markdown(
    "Heiner (1983) argued that **decision errors**, not optimisation, shape behaviour. When the difficulty "
    "of a problem exceeds an agent's competence, the agent faces genuine uncertainty. Its attempts to "
    "exploit flexibility then produce mistakes, and **constraining behaviour to a simple rule can "
    "outperform flexible 'optimising' behaviour, even when flexibility is free.** This app turns the "
    "paper's cobweb-market experiments into an agent-based laboratory for testing that claim.")

st.header("1 · The theory in five quantities", divider="gray")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown(
        r"""
**Uncertainty is a gap.** On a common scale, the agent's *competence* is compared with the task's *difficulty*:

$$\text{CD-gap} = \max\{0,\ \text{Difficulty} - \text{Competence}\}$$

**A rule and its exceptions.** The agent follows a simple rule **B** (here: *keep last period's production*).
Because B is not optimal there are **preferred exceptions**, moments where deviating is better.

| Symbol | Meaning |
|---|---|
| $\pi$ | probability that a preferred exception exists |
| $r(U)$ | P(deviate \| preferred exception) = 1 − P(type I error) |
| $w(U)$ | P(deviate \| no preferred exception) = P(type II error) |
| $G$ | average gain from deviating at the right time |
| $D$ | average loss from deviating at the wrong time |

As uncertainty $U$ grows, $r$ falls and $w$ rises.
""")
with c2:
    st.markdown(
        r"""
**When does flexibility pay?** Gains must accumulate faster than losses:

$$\pi\, r\, G \;>\; (1-\pi)\, w\, D$$

which rearranges to Heiner's **reliability condition**:

$$\underbrace{\frac{r(U)}{w(U)}}_{\text{reliability ratio}} \;>\; \underbrace{\frac{D}{G}\cdot\frac{1-\pi}{\pi}}_{\text{tolerance limit}}$$

If the condition fails, **more behavioural flexibility is harmful**, and the agent does better by sticking to B
even though B is sometimes wrong. The condition does not require the agent to *know* r, w, π, G or D.
""")

st.header("2 · Interactive reliability calculator", divider="gray")
st.caption("Illustrative functional forms: r(U) = 1 − (1 − r∞)(1 − e^(−U/s)),  w(U) = w∞(1 − e^(−U/s)). "
           "Move the sliders to see where the reliability ratio drops below the tolerance limit.")
cc = st.columns(4)
pi = cc[0].slider("π (chance of a preferred exception)", 0.05, 0.95, 0.35, 0.01,
                   help="Probability that, at a given decision, deviating from rule B would actually be better.")
G = cc[1].slider("G (gain from a correct deviation)", 1.0, 100.0, 30.0, 1.0,
                  help="Average payoff gained by deviating from B when a preferred exception exists.")
D = cc[2].slider("D (loss from a mistaken deviation)", 1.0, 100.0, 20.0, 1.0,
                  help="Average payoff lost by deviating from B when no preferred exception exists.")
U = cc[3].slider("Current uncertainty U (CD-gap)", 0.0, 10.0, 2.0, 0.1,
                  help="Gap between the difficulty of the problem and the agent's competence. U = 0 means a "
                       "perfectly reliable agent (r = 1, w = 0).")
cc = st.columns(4)
r_inf = cc[0].slider("r∞ (right-deviation rate at extreme U)", 0.0, 1.0, 0.5, 0.01,
                      help="Limit of r(U), the probability of deviating when an exception exists, as U → ∞. "
                           "r(U) falls from 1 toward r∞.")
w_inf = cc[1].slider("w∞ (wrong-deviation rate at extreme U)", 0.0, 1.0, 0.5, 0.01,
                      help="Limit of w(U), the probability of deviating when no exception exists, as U → ∞. "
                           "w(U) rises from 0 toward w∞. r∞ = w∞ means pure guessing.")
s = cc[2].slider("s (how fast competence is overwhelmed)", 0.2, 10.0, 2.0, 0.1,
                  help="Scale of uncertainty in e^(−U/s). Small s: r and w reach their limits quickly as U rises; "
                       "large s: the agent stays reliable longer.")
cost_f = cc[3].slider("Cost of flexibility (adds to D, subtracts from G)", 0.0, 30.0, 0.0, 0.5,
                       help="Price of being able to deviate. Heiner's point holds even at 0: flexibility can hurt "
                            "when it is free.")

Gf, Df = max(G - cost_f, 1e-6), D + cost_f
us = np.linspace(0.0, 10.0, 401)
r_u = 1 - (1 - r_inf) * (1 - np.exp(-us / s))
w_u = w_inf * (1 - np.exp(-us / s))
ratio = np.where(w_u > 0, r_u / np.maximum(w_u, 1e-12), np.inf)
tol = Df / Gf * (1 - pi) / pi
r_now = 1 - (1 - r_inf) * (1 - np.exp(-U / s))
w_now = w_inf * (1 - np.exp(-U / s))
ratio_now = r_now / w_now if w_now > 0 else np.inf
net = pi * r_now * Gf - (1 - pi) * w_now * Df

cross = us[np.argmax(ratio < tol)] if np.any(ratio < tol) else None
m = st.columns(5)
m[0].metric("r(U)", f"{r_now:.2f}")
m[1].metric("w(U)", f"{w_now:.2f}")
m[2].metric("Reliability ratio", "∞" if np.isinf(ratio_now) else f"{ratio_now:.2f}")
m[3].metric("Tolerance limit", f"{tol:.2f}")
m[4].metric("Net gain per decision", f"{net:+.2f}")
if ratio_now > tol:
    st.success(f"Reliability condition **holds** at U = {U:.1f}: flexibility pays.", icon="✅")
else:
    st.error(f"Reliability condition **fails** at U = {U:.1f}: the agent should constrain behaviour to rule B.",
             icon="❌")

fig = go.Figure()
yv = np.clip(ratio, 1e-2, 1e3)
fig.add_trace(go.Scatter(x=us, y=yv, name="Reliability ratio r/w", line=dict(color=CAT[0], width=2),
                         hovertemplate="U %{x:.2f}<br>r/w %{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=us, y=np.full_like(us, tol), name="Tolerance limit", line=dict(color=CAT[1], width=2,
                         dash="dash"), hovertemplate="tolerance %{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[U], y=[min(max(ratio_now, 1e-2), 1e3)], mode="markers", name="Current U",
                         marker=dict(size=11, color=CAT[0], line=dict(color="white", width=2))))
if cross is not None:
    fig.add_vrect(x0=cross, x1=10, fillcolor="rgba(227,73,72,0.08)", line_width=0)
    fig.add_annotation(x=(cross + 10) / 2, y=np.log10(1e3) - 0.2, yref="y", text="constrained behaviour wins",
                       showarrow=False, font=dict(size=12))
fig.update_yaxes(type="log", title="ratio (log scale)")
fig.update_xaxes(title="Uncertainty U = CD-gap")
st.plotly_chart(style(fig, 360))

st.header("3 · How the simulation operationalises the theory", divider="gray")
st.markdown(
    """
| Heiner's concept | In the cobweb market |
|---|---|
| **Difficulty** | Raw-material cost volatility Δ: the cost moves by up to ±Δ per period and firms must commit to production *before* the new cost is known. The paper uses the serial correlation of cost as the uncertainty proxy. |
| **Competence** | The firm's cost foresight κ (share of the coming change it anticipates) and perception noise σ. The paper's firms have κ = 0, σ = 0. |
| **Measured CD-gap** | RMSE of each firm's cost perception, measured during the run. |
| **Rule B** | *Keep producing last period's quantity* (the VBA "Never" rule). |
| **The flexible alternative** | The Cournot best reply (partial adjustment weight φ) or the Bertrand margin-feedback rule (sensitivity φ). |
| **Behavioural flexibility** | φ (how far a firm moves) and the **selection rule** (when it is allowed to deviate from B: Always, SR1 *Small*, SR2 *Large*, or *Adaptive*). |
| **Preferred exception** | A period in which adopting the recommendation earns more than following B, evaluated by forking the market (horizon H). |
| **π, r, w, G, D** | Counted for every firm and every decision, so the reliability condition is computed per firm. |
| **Performance** | Average profit per period after burn-in, compared across firms of different flexibility, or against the firm's own rigid twin. |
""")

st.header("4 · Competing hypotheses", divider="gray")
st.dataframe(pd.DataFrame([
    ("Flexibility is free (a = b = 0)", "Flexibility can still hurt: errors limit its value",
     "More flexibility never hurts (relaxing a constraint cannot lower the optimum)", "Hypothesis tests → H1"),
    ("Industry profitability rises", "Mistakes become cheaper, so flexibility becomes favoured (a switch point)",
     "Also compatible: larger rewards to flexibility", "H2"),
    ("Volatility (difficulty) rises", "Reliability falls, so flexibility is punished more",
     "More volatility raises the value of flexibility", "H3"),
    ("Fixed costs rise, same marginal costs", "Gains shrink and losses grow, so the switch point moves up",
     "No effect: decisions are made at the margin", "H4"),
    ("Competition intensifies (m* ↓)", "The paper's observation: the switch happens at lower profit levels", "n/a", "H5"),
    ("Competence rises (κ ↑)", "The CD-gap narrows: r ↑, w ↓, flexibility pays again", "Information always helps",
     "CD-gap explorer"),
    ("Firms can choose their own flexibility", "Volatile industries evolve toward rigidity",
     "Volatile industries evolve toward flexibility", "Endogenous flexibility"),
    ("Firms satisfying the RC", "…should beat their rigid selves; violators should not", "Flexibility always wins",
     "Does the RC predict performance?"),
], columns=["Situation", "Reliability condition (Heiner)", "Traditional / optimising view", "Where to test"]),
    hide_index=True, width="stretch")

st.header("5 · Extending the theory", divider="gray")
st.markdown(
    r"""
Three extensions turn the paper's test into a contribution to the uncertainty literature. Each has its own page.

**A dynamic reliability condition** (*Dynamic RC* page). Heiner's 1983 condition values each deviation as a
one-shot bet. In markets a decision **persists** and **provokes reactions**, so its value is
$\text{immediate} + \text{persistence} + \text{strategic feedback}$, measured over a horizon $H$ with discount $\gamma$.
The simulation shows the one-period RC, which is what the paper proposed, is uninformative about who benefits from
flexibility, while the dynamic RC predicts it. Most of the missing value is strategic feedback.

**Heiner (1989), partial adjustment toward an imperfectly perceived target.** The paper's Cournot rule is
exactly Heiner's later model: move a fraction $\beta$ of the way toward the perceived best reply
$\hat x_t = x^*_t + \xi_t$. The remaining gap is $d_t = (1-\beta)\Delta^*_t - \beta\xi_t$, a trade-off between lagging
the target and importing errors. Theorem 2 gives the maximal reliable speed
$\beta_0 = 1/[(1+K)(1-f')]$ from the error-to-signal ratio $K$. The simulation measures $K$ and finds the
profit-maximising $\beta$ at or below $\beta_0$, with the market collapsing beyond the stability limit $4/(n+1)$.

**Risk versus Knightian uncertainty** (*Risk vs Knightian uncertainty* page). Unannounced demand-regime
shifts make model-based firms' demand model wrong, not just noisy. At matched unpredictability, flexibility becomes
harmful for model-based (Cournot) firms but stays valuable for model-free (Bertrand) firms, and only the dynamic RC
registers the change. Around each shift, firms show Heiner's *punctuated* slow–quick–slow adjustment.

**Out-of-sample testing** (*Does the RC predict performance?* page). The RC is estimated in the first part of each
run and used to predict flexible-vs-rigid performance in the second part, scored by AUC. This is the
signal-detection statistic matching Heiner's reading of r and w as hit and false-alarm rates.
""")
