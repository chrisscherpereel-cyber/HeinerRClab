"""One interactive illustration per theory, each a small, self-contained demonstration of its core mechanism."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.common import CAT, style


# ------------------------------------------------------------------------------------------------ Heiner
def heiner():
    st.caption("Illustrative functional forms: r(U) = 1 − (1 − r∞)(1 − e^(−U/s)),  w(U) = w∞(1 − e^(−U/s)). "
               "Move the sliders to see where the reliability ratio drops below the tolerance limit.")
    cc = st.columns(4)
    pi = cc[0].slider("π (chance of a preferred exception)", 0.05, 0.95, 0.35, 0.01, key="th_h_pi",
                      help="Probability that, at a given decision, deviating from rule B would actually be better.")
    G = cc[1].slider("G (gain from a correct deviation)", 1.0, 100.0, 30.0, 1.0, key="th_h_G",
                     help="Average payoff gained by deviating from B when a preferred exception exists.")
    D = cc[2].slider("D (loss from a mistaken deviation)", 1.0, 100.0, 20.0, 1.0, key="th_h_D",
                     help="Average payoff lost by deviating from B when no preferred exception exists.")
    U = cc[3].slider("Current uncertainty U (CD-gap)", 0.0, 10.0, 2.0, 0.1, key="th_h_U",
                     help="Gap between the difficulty of the problem and the agent's competence. U = 0 means a "
                          "perfectly reliable agent (r = 1, w = 0).")
    cc = st.columns(4)
    r_inf = cc[0].slider("r∞ (right-deviation rate at extreme U)", 0.0, 1.0, 0.5, 0.01, key="th_h_rinf",
                         help="Limit of r(U), the probability of deviating when an exception exists, as U → ∞.")
    w_inf = cc[1].slider("w∞ (wrong-deviation rate at extreme U)", 0.0, 1.0, 0.5, 0.01, key="th_h_winf",
                         help="Limit of w(U), the probability of deviating when no exception exists, as U → ∞. "
                              "r∞ = w∞ means pure guessing.")
    s = cc[2].slider("s (how fast competence is overwhelmed)", 0.2, 10.0, 2.0, 0.1, key="th_h_s",
                     help="Scale of uncertainty in e^(−U/s). Small s: r and w reach their limits quickly as U rises.")
    cost_f = cc[3].slider("Cost of flexibility (adds to D, subtracts from G)", 0.0, 30.0, 0.0, 0.5, key="th_h_cost",
                          help="Price of being able to deviate. The theory's point holds even at 0: flexibility can "
                               "hurt when it is free.")
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
        st.error(f"Reliability condition **fails** at U = {U:.1f}: the agent should constrain behavior to rule B.",
                 icon="❌")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=us, y=np.clip(ratio, 1e-2, 1e3), name="Reliability ratio r/w",
                             line=dict(color=CAT[0], width=2)))
    fig.add_trace(go.Scatter(x=us, y=np.full_like(us, tol), name="Tolerance limit",
                             line=dict(color=CAT[1], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[U], y=[min(max(ratio_now, 1e-2), 1e3)], mode="markers", name="Current U",
                             marker=dict(size=11, color=CAT[0], line=dict(color="white", width=2))))
    if cross is not None:
        fig.add_vrect(x0=cross, x1=10, fillcolor="rgba(227,73,72,0.08)", line_width=0)
    fig.update_yaxes(type="log", title="ratio (log scale)")
    fig.update_xaxes(title="Uncertainty U = CD-gap")
    st.plotly_chart(style(fig, 340))


# ------------------------------------------------------------------------------------------------ optimization
def _ewma_mse(alpha, Q, R):
    """Steady-state squared error of exponential smoothing of a random walk (step variance Q) observed with noise R."""
    return ((1 - alpha) ** 2 * Q + alpha ** 2 * R) / (alpha * (2 - alpha))


def optimiser():
    st.caption("A target moves as a random walk and is observed with noise. Exponential smoothing with gain α "
               "tracks it; the optimal (Kalman) gain balances lagging the target against importing noise.")
    c = st.columns(2)
    q = c[0].slider("Target volatility (s.d. of each step)", 0.1, 10.0, 2.0, 0.1, key="th_o_q",
                    help="How much the true target moves each period.")
    r = c[1].slider("Observation noise (s.d.)", 0.0, 20.0, 5.0, 0.5, key="th_o_r",
                    help="Noise in each observation of the target.")
    Q, R = q ** 2, max(r ** 2, 1e-9)
    al = np.linspace(0.01, 1.0, 300)
    mse = _ewma_mse(al, Q, R)
    pm = (Q + np.sqrt(Q ** 2 + 4 * Q * R)) / 2
    k = pm / (pm + R)
    m = st.columns(3)
    m[0].metric("Optimal gain K", f"{k:.2f}")
    m[1].metric("Error at K", f"{np.sqrt(_ewma_mse(k, Q, R)):.2f}")
    m[2].metric("Error with full adjustment (α = 1)", f"{np.sqrt(_ewma_mse(1.0, Q, R)):.2f}")
    fig = go.Figure(go.Scatter(x=al, y=np.sqrt(mse), line=dict(color=CAT[0], width=2), name="Tracking error"))
    fig.add_vline(x=k, line=dict(color=CAT[1], dash="dash"), annotation_text="Kalman gain")
    fig.update_xaxes(title="Gain α (share of each surprise acted on)")
    fig.update_yaxes(title="Root-mean-square tracking error")
    st.plotly_chart(style(fig, 320))
    st.caption("More noise relative to target volatility lowers the optimal gain: an optimizer attenuates its "
               "response to noisy signals (Muth 1960; Kalman 1960), without any fixed rule.")


# ------------------------------------------------------------------------------------------------ real options
def options():
    st.caption("A firm tracks a randomly moving target. Every period it loses the squared gap to the target; every "
               "adjustment costs a fixed amount. The best policy is an inaction band.")
    c = st.columns(2)
    sig = c[0].slider("Volatility of the target (s.d. per period)", 0.2, 5.0, 1.0, 0.1, key="th_r_sig",
                      help="How much the target moves each period.")
    k = c[1].slider("Fixed cost of each adjustment", 0.0, 200.0, 40.0, 5.0, key="th_r_k",
                    help="Cost paid every time the firm moves to the target.")
    rng = np.random.default_rng(1)
    steps = rng.standard_normal(4000)
    bands = np.linspace(0.0, 12.0, 49)

    def costs_for(s):
        """Average cost per period for every band at once (vectorized over bands)."""
        gap, total, n = np.zeros(len(bands)), np.zeros(len(bands)), np.zeros(len(bands))
        for z in steps * s:
            gap += z
            move = np.abs(gap) > bands
            gap[move] = 0.0
            n += move
            total += gap ** 2
        return (total + k * n) / len(steps)

    costs = costs_for(sig)
    b_star = bands[costs.argmin()]
    sigs = [0.5, 1.0, 2.0, 4.0]
    bstars = [bands[costs_for(s_).argmin()] for s_ in sigs]
    m = st.columns(2)
    m[0].metric("Best inaction band", f"{b_star:.2f}")
    m[1].metric("Adjust every period instead: cost increase", f"{costs[0] - costs.min():+.1f}")
    fig = go.Figure(go.Scatter(x=bands, y=costs, line=dict(color=CAT[0], width=2), name="Average cost"))
    fig.add_vline(x=b_star, line=dict(color=CAT[1], dash="dash"), annotation_text="best band")
    fig.update_xaxes(title="Inaction band (adjust only when the gap exceeds it)")
    fig.update_yaxes(title="Average cost per period")
    st.plotly_chart(style(fig, 320))
    st.caption("Best band at volatility " + ", ".join(f"{s_:g}: {b:.1f}" for s_, b in zip(sigs, bstars))
               + ". The band widens with volatility (hysteresis; Dixit 1989).")


# ------------------------------------------------------------------------------------------------ cobweb
def cobweb():
    st.caption("Producers commit output on an expected price; the market then clears on demand. Adaptive "
               "expectations move the forecast a share λ toward the last price.")
    c = st.columns(3)
    ratio = c[0].slider("Supply slope ÷ demand slope (b/d)", 0.2, 3.0, 1.2, 0.05, key="th_c_ratio",
                        help="How strongly output reacts to the expected price, relative to how strongly demand "
                             "reacts to the price.")
    lam = c[1].slider("Adaptive-expectations gain λ", 0.05, 1.0, 1.0, 0.05, key="th_c_lam",
                      help="Share of the last forecast error added to the expected price (λ = 1: naive "
                           "expectations).")
    p0 = c[2].slider("Initial expected price (equilibrium = 10)", 5.0, 15.0, 12.0, 0.5, key="th_c_p0",
                     help="Where expectations start, relative to the equilibrium price 10.")
    d = 1.0
    b = ratio * d
    a = 10 * d + (10 * b)          # equilibrium price 10 with c = 0
    pe, prices = p0, []
    for _ in range(40):
        p = (a - b * pe) / d
        prices.append(p)
        pe = pe + lam * (p - pe)
    eig = 1 - lam * (1 + b / d)
    stable = abs(eig) < 1
    m = st.columns(2)
    m[0].metric("Root of the expectation dynamics", f"{eig:+.2f}")
    m[1].metric("Stability limit (2 − λ)/λ", f"{(2 - lam) / lam:.2f}")
    (st.success if stable else st.error)(
        f"b/d = {ratio:.2f} {'<' if stable else '≥'} (2 − λ)/λ: the market {'converges' if stable else 'diverges'}.")
    fig = go.Figure(go.Scatter(y=np.clip(prices, -50, 70), mode="lines+markers", line=dict(color=CAT[0], width=2),
                               name="Price"))
    fig.add_hline(y=10, line=dict(color="rgba(128,128,128,0.8)", dash="dot"))
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Price")
    st.plotly_chart(style(fig, 300))


# ------------------------------------------------------------------------------------------------ heuristics
def heuristic():
    st.caption("Predict an outcome from four cues. A fitted linear model (flexible) estimates every weight; a "
               "tallying rule (simple) gives every cue the same weight. Both are scored on new cases.")
    c = st.columns(2)
    noise = c[0].slider("Noise in the outcome (s.d.)", 0.0, 5.0, 2.0, 0.1, key="th_b_noise",
                        help="Unpredictable part of the outcome.")
    spread = c[1].slider("How unequal the true weights are", 0.0, 1.0, 0.3, 0.05, key="th_b_spread",
                         help="0 = all cues matter equally (tallying is unbiased); 1 = very unequal weights.")
    rng = np.random.default_rng(7)
    w = 1 + spread * np.array([1.5, 0.5, -0.5, -0.9])
    ns = [5, 8, 12, 20, 40, 80, 160]
    fit, tally = [], []
    for n in ns:
        f_, t_ = [], []
        for _ in range(200):
            X = rng.standard_normal((n, 4))
            y = X @ w + noise * rng.standard_normal(n)
            Xt = rng.standard_normal((400, 4))
            yt = Xt @ w + noise * rng.standard_normal(400)
            beta = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0]
            f_.append(np.corrcoef(np.column_stack([np.ones(400), Xt]) @ beta, yt)[0, 1])
            t_.append(np.corrcoef(Xt.sum(1), yt)[0, 1])
        fit.append(np.mean(f_))
        tally.append(np.mean(t_))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=fit, mode="lines+markers", name="Fitted weights (flexible)",
                             line=dict(color=CAT[0], width=2)))
    fig.add_trace(go.Scatter(x=ns, y=tally, mode="lines+markers", name="Equal weights (simple)",
                             line=dict(color=CAT[1], width=2)))
    fig.update_xaxes(title="Cases available for learning", type="log")
    fig.update_yaxes(title="Accuracy on new cases (correlation)")
    st.plotly_chart(style(fig, 320))
    better = [n for n, f_, t_ in zip(ns, fit, tally) if t_ > f_]
    st.caption(("The simple rule predicts better with up to " + str(max(better)) + " cases"
                if better else "The fitted model predicts better at every sample size here")
               + ". Less data, more noise and more equal true weights all favor the simple rule (Dawes 1979; "
                 "Gigerenzer & Brighton 2009).")


# ------------------------------------------------------------------------------------------------ satisficing
def satisficing():
    st.caption("A firm chooses among nine options of unknown value. It keeps its option while profit meets an "
               "aspiration that adapts to experience, and tries a neighboring option when profit falls short.")
    c = st.columns(2)
    noise = c[0].slider("Noise in observed profit (s.d.)", 0.0, 5.0, 1.0, 0.1, key="th_s_noise",
                        help="How much profit varies around an option's true value.")
    alpha = c[1].slider("Aspiration adaptation speed α", 0.01, 0.5, 0.1, 0.01, key="th_s_alpha",
                        help="How quickly the aspiration moves toward recent profit.")
    rng = np.random.default_rng(3)
    value = 10 - 0.5 * (np.arange(9) - 6.0) ** 2 / 4
    T = 300
    x, A = 0, value[0]
    prof, asp, opt = [], [], []
    for t in range(T):
        p = value[x] + noise * rng.standard_normal()
        prof.append(p)
        asp.append(A)
        opt.append(x)
        if p < A:
            x = int(np.clip(x + rng.choice([-1, 1]), 0, 8))
        A = A + alpha * (p - A)
    searches = int(np.sum(np.diff(opt) != 0))
    m = st.columns(3)
    m[0].metric("Option reached (best = 6)", f"{opt[-1]}")
    m[1].metric("Changes of option", f"{searches}")
    m[2].metric("Mean true value, last 100 periods", f"{np.mean(value[np.array(opt[-100:])]):.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prof, name="Profit", line=dict(color=CAT[0], width=1)))
    fig.add_trace(go.Scatter(y=asp, name="Aspiration", line=dict(color=CAT[1], width=2)))
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Profit")
    st.plotly_chart(style(fig, 300))
    st.caption("More noise pushes profit below aspiration more often, so the firm searches more, including away "
               "from good options (Cyert & March 1963).")


# ------------------------------------------------------------------------------------------------ reinforcement
def rl():
    st.caption("A learner repeatedly picks one of two actions. Their payoffs swap every S periods. Softmax learning "
               "with learning rate η and temperature τ; the curve shows the share of choices of the currently "
               "better action, averaged over 200 learners.")
    c = st.columns(3)
    eta = c[0].slider("Learning rate η", 0.01, 1.0, 0.1, 0.01, key="th_l_eta",
                      help="How strongly the latest payoff updates the value of the chosen action.")
    tau = c[1].slider("Temperature τ", 0.05, 3.0, 0.5, 0.05, key="th_l_tau",
                      help="Exploration: high τ chooses more randomly.")
    S = c[2].slider("Periods between payoff swaps S", 20, 400, 100, 10, key="th_l_S",
                    help="How often the environment changes.")
    rng = np.random.default_rng(5)
    T, N = 400, 200
    V = np.zeros((N, 2))
    good = np.zeros(T)
    for t in range(T):
        best = (t // S) % 2
        z = V / tau
        p1 = 1 / (1 + np.exp(z[:, 0] - z[:, 1]))
        a = (rng.random(N) < p1).astype(int)
        r = np.where(a == best, 1.0, 0.0) + 0.5 * rng.standard_normal(N)
        V[np.arange(N), a] += eta * (r - V[np.arange(N), a])
        good[t] = (a == best).mean()
    fig = go.Figure(go.Scatter(y=good, line=dict(color=CAT[0], width=2), name="Share choosing the better action"))
    for k in range(S, T, S):
        fig.add_vline(x=k, line=dict(color="rgba(128,128,128,0.5)", dash="dot"))
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Share choosing the better action", range=[0, 1])
    st.plotly_chart(style(fig, 300))
    st.metric("Average share choosing the better action", f"{good.mean():.2f}")
    st.caption("Fast learning tracks changes but is noisy; slow learning is stable but lags each change. Exploration "
               "costs choices when nothing changes and pays when something does (March 1991; Sutton & Barto 2018).")


# ------------------------------------------------------------------------------------------------ imitation
def imitation():
    st.caption("Firms in a Cournot market with linear demand P = 100 − Q and unit cost 10. Each period every firm "
               "copies the output of last period's most profitable firm, with small experimentation.")
    c = st.columns(2)
    n = c[0].slider("Number of firms", 2, 10, 4, 1, key="th_i_n", help="Firms in the market.")
    eps = c[1].slider("Experimentation (s.d. of output changes)", 0.1, 5.0, 1.0, 0.1, key="th_i_eps",
                      help="Random variation added to each firm's output every period.")
    rng = np.random.default_rng(11)
    T = 400
    q = np.full(n, 90.0 / (n + 1)) + rng.normal(0, 2, n)
    total = []
    for _ in range(T):
        P = max(0.0, 100 - q.sum())
        prof = (P - 10) * q
        q = np.maximum(0.0, q[prof.argmax()] + eps * rng.standard_normal(n))
        total.append(q.sum())
    fig = go.Figure(go.Scatter(y=total, line=dict(color=CAT[0], width=2), name="Market output"))
    for y, lab, col in ((90 * n / (n + 1), "Cournot–Nash", CAT[1]), (90.0, "Walrasian (competitive)", CAT[2]),
                        (45.0, "Collusive", CAT[3])):
        fig.add_hline(y=y, line=dict(color=col, dash="dash"), annotation_text=lab, annotation_position="top left")
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Total output")
    st.plotly_chart(style(fig, 320))
    st.caption("Imitating the most profitable firm pushes output above the Cournot–Nash level toward the competitive "
               "level: when price exceeds cost, the firm that produces most earns most, so it is copied "
               "(Vega-Redondo 1997; Huck et al. 1999).")


ILLUSTRATIONS = dict(heiner=heiner, optimiser=optimiser, options=options, cobweb=cobweb, heuristic=heuristic,
                     satisficing=satisficing, rl=rl, imitation=imitation)
