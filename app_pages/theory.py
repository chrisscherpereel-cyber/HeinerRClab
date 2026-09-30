import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.literature import CONTRIBUTIONS, HYPOTHESES, cite
from ui.common import CAT, style

st.title("Heiner's reliability condition")
st.markdown(
    "Heiner (1983) argued that **decision errors**, not optimisation, shape behaviour. When the difficulty "
    "of a problem exceeds an agent's competence, the agent faces genuine uncertainty. Its attempts to "
    "exploit flexibility then produce mistakes, and **constraining behaviour to a simple rule can "
    "outperform flexible 'optimising' behaviour, even when flexibility is free.** This app is an agent-based "
    "laboratory, built on a cobweb oligopoly market, that tests that claim against its main rivals.")

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
| **Difficulty** | Raw-material cost volatility Δ: the cost moves by up to ±Δ per period and firms must commit to production *before* the new cost is known. The serial correlation of cost is reported as an empirical predictability proxy. |
| **Competence** | The firm's cost foresight κ (share of the coming change it anticipates) and perception noise σ. Baseline firms have κ = 0, σ = 0. |
| **Measured CD-gap** | RMSE of each firm's cost perception, measured during the run. |
| **Rule B** | *Keep producing last period's quantity* (the "Never" selection rule). |
| **The flexible alternative** | The Cournot best reply (partial adjustment weight φ) or the Bertrand margin-feedback rule (sensitivity φ). |
| **Behavioural flexibility** | φ (how far a firm moves) and the **selection rule** (when it is allowed to deviate from B: Always, SR1 *Small*, SR2 *Large*, or *Adaptive*). |
| **Preferred exception** | A period in which adopting the recommendation earns more than following B, evaluated by forking the market (horizon H). |
| **π, r, w, G, D** | Counted for every firm and every decision, so the reliability condition is computed per firm. |
| **Performance** | Average profit per period after burn-in, compared across firms of different flexibility, or against the firm's own rigid twin. |
""")

st.header("4 · Hypotheses, alternatives and the research behind them", divider="gray")
st.dataframe(pd.DataFrame([{
    "ID": h.hid, "Hypothesis": h.title, "Reliability condition predicts": h.rc_prediction,
    "Alternative predicts": f"{h.alt_label}: {h.alt_prediction}",
    "Supporting research": cite(*[k for k, _ in h.support]),
    "Alternative research": cite(*[k for k, _ in h.alternative]), "Where to test": h.where} for h in HYPOTHESES]),
    hide_index=True, width="stretch")
st.caption("H1–H12 are on the *Hypothesis tests* page; the other hypotheses have their own pages. Every hypothesis "
           "card in the app opens its research basis. The *Competing theories* page pits the reliability condition "
           "against its rivals in a tournament of experiments and an out-of-sample forecasting horse race, and the "
           "*Research & contribution* page collects the full bibliography.")

st.header("5 · Contribution to the literature", divider="gray")
for k, (title, text, keys) in enumerate(CONTRIBUTIONS, 1):
    st.markdown(f"**C{k} · {title}.** {text} *({cite(*keys)})*")
