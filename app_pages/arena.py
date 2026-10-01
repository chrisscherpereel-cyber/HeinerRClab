from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.arena import AGENTS, KEYS, PREREG, QUICK, Prereg, replicate, run_protocol
from heiner_abm.literature import cite
from ui.common import CAT, DIVERGING, download, hypothesis_card, style, verdict

st.title("Agent tournament: rival theories as competing agents")
st.caption("Each theory of flexibility under uncertainty is implemented as a decision rule. The rules compete in the "
           "same cobweb market, with the same information and the same random shocks, under a fairness protocol "
           "fixed in advance. This replaces predictions written by the modeller with behaviour that can win or lose.")

hypothesis_card(
    "ARENA",
    "Nine agent types share one market: the reliability-condition agent, a filtered best reply (optimisation), an "
    "inaction band (real options), adaptive expectations (cobweb theory), win-stay/lose-shift (simple heuristics), "
    "aspiration-level search (satisficing), a reinforcement learner, an imitator, and rule B as a benchmark. The "
    "reliability-condition agent, the optimiser and the inaction band chase the **same filtered target**; they differ "
    "only in *when* they act, which isolates Heiner's selection rule.")

st.header("1 · The fairness protocol", divider="gray")
st.markdown(
    "1. **Same market, same information.** Every agent sees last period's price and market quantity, its own output "
    "and a cost estimate of the same quality. Model-based agents use the same (possibly outdated) demand model.\n"
    "2. **Equal tuning budget.** Every agent's free parameters are tuned with the same number of candidate settings "
    "(Latin-hypercube search, defaults included) on *training* environments, in two rounds: round 2 tunes against the "
    "round-1 tuned rivals.\n"
    "3. **Held-out, randomly drawn environments.** Test environments are drawn from pre-registered ranges with "
    "different seeds and never used for tuning.\n"
    "4. **Several yardsticks.** Head-to-head profit in mixed markets, invasion tests (can one mutant out-earn a "
    "population of another type?), global sensitivity analysis over the environment space, and replication of the "
    "whole protocol with fresh seeds.\n"
    "5. **A frozen plan.** Ranges, budgets, hypotheses and decision rules, together with the agent and analysis code, "
    "are hashed. A run with any change is labelled exploratory.")
st.caption("Methods: " + cite("axelrod1984", "maynardsmith1973", "mckay1979", "bergstra2012", "holm1979",
                               "saltelli2008", "nosek2018"))

st.subheader("The agents")
st.dataframe(pd.DataFrame([dict(Agent=a.name, Theory=a.theory, Rule=a.rule,
                                Parameters=", ".join(f"{n} ∈ [{s.lo:g}, {s.hi:g}]" for n, s in a.SPACE.items()) or "–",
                                Sources=cite(*a.sources)) for a in AGENTS.values()]),
             hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ plan
st.header("2 · Pre-registered plan", divider="gray")
scale = st.radio("Protocol", ["Pre-registered", "Quick check", "Custom"], horizontal=True, key="arena_scale",
                 help="Pre-registered runs the frozen plan (about 35 seconds). Quick check is a small exploratory run "
                      "for trying things out. Custom lets you change the settings; the run is then exploratory.")
pr = PREREG if scale == "Pre-registered" else QUICK
if scale == "Custom":
    c = st.columns(4)
    budget = c[0].number_input("Tuning budget per agent", 4, 200, PREREG.budget,
                               help="Candidate parameter settings evaluated for every agent (equal for all).")
    n_train = c[1].number_input("Training environments", 4, 200, PREREG.n_train,
                                help="Environments used only for tuning.")
    n_test = c[2].number_input("Test environments", 4, 300, PREREG.n_test,
                               help="Held-out environments used for every result.")
    periods = c[3].number_input("Periods per market", 150, 5000, PREREG.periods, step=50,
                                help="Length of every simulated market.")
    c = st.columns(4)
    rounds = c[0].number_input("Tuning rounds", 1, 4, PREREG.rounds,
                               help="Round r tunes each agent against the others' round r − 1 settings.")
    residents = c[1].number_input("Residents in invasion tests", 2, 10, PREREG.residents,
                                  help="Firms of the resident type; one mutant joins them.")
    train_seed = c[2].number_input("Training seed", 0, 10**6, PREREG.train_seed, help="Seed for training environments.")
    test_seed = c[3].number_input("Test seed", 0, 10**6, PREREG.test_seed, help="Seed for test environments.")
    pr = Prereg(**{**asdict(PREREG), "budget": int(budget), "n_train": int(n_train), "n_test": int(n_test),
                   "periods": int(periods), "rounds": int(rounds), "residents": int(residents),
                   "train_seed": int(train_seed), "test_seed": int(test_seed)})
exploratory = pr.digest != PREREG.digest
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{pr.digest}` " + ("· **exploratory** (differs from the registered plan "
                                                     f"`{PREREG.digest}`)" if exploratory else "· registered plan"))
    st.caption(f"{pr.n_train} training and {pr.n_test} × {pr.reps_test} test environments · {pr.periods} periods "
               f"(burn-in {pr.burn_in}) · budget {pr.budget} per agent × {pr.rounds} rounds · invasion: "
               f"{pr.residents} residents + 1 mutant · α = {pr.alpha}")
with c2:
    st.download_button("Download plan (.json)", pr.to_json().encode(), file_name=f"prereg_{pr.digest}.json",
                       mime="application/json", width="stretch",
                       help="File this with a registry (for example OSF or AsPredicted) before running, so the "
                            "registration is time-stamped by a third party.")
st.dataframe(pd.DataFrame(pr.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")
st.caption("Environment ranges: " + "; ".join(f"{k} {lo:g}–{hi:g}" for k, (lo, hi) in pr.env_ranges
                                              if k != "share_with_shifts")
           + f"; demand-regime shifts in {pr.ranges['share_with_shifts'][0]:.0%} of environments.")

store = st.session_state.setdefault("arena_results", {})
if st.button("Run tournament", type="primary", key="arena_run"):
    bar = st.progress(0.0, "Starting")
    store[pr.digest] = run_protocol(pr, lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
res = store.get(pr.digest)
if res is None:
    st.info("Run the tournament to see results for this plan.", icon="ℹ️")
    st.stop()

# ------------------------------------------------------------------------------------------------ results
st.header("3 · Pre-registered hypotheses", divider="gray")
if res.exploratory:
    st.warning("Exploratory run: the plan differs from the registered one, so these verdicts are not confirmatory.",
               icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject",
            f"**{r['id']}** · {r['hypothesis']} **{r['verdict'].capitalize()}**: {r['result']}.")

st.header("4 · Tuning with an equal budget", divider="gray")
log = res.tuning_log
last = log[log["round"] == log["round"].max()]
st.dataframe(last[["agent", "default_score", "best_score", "gain"] + [c for c in last if c.startswith("param:")]],
             hide_index=True, width="stretch", column_config={
                 "agent": "Agent", "default_score": st.column_config.NumberColumn("Profit, default settings", format="%.0f"),
                 "best_score": st.column_config.NumberColumn("Profit, tuned", format="%.0f"),
                 "gain": st.column_config.NumberColumn("Gain from tuning", format="%+.0f")})
st.caption(f"Final round, training environments only. Every agent evaluated {pr.budget} settings per round.")

st.header("5 · Head-to-head in held-out mixed markets", divider="gray")
rk = res.ranking
fig = go.Figure(go.Bar(
    y=rk["agent"][::-1], x=rk["rel_profit"][::-1], orientation="h",
    marker_color=[CAT[0] if k == "heiner" else "#9aa0a6" for k in rk["key"][::-1]],
    error_x=dict(type="data", symmetric=False, array=(rk["hi"] - rk["rel_profit"])[::-1],
                 arrayminus=(rk["rel_profit"] - rk["lo"])[::-1], thickness=1.2, width=4),
    hovertemplate="%{y}<br>profit vs market mean %{x:,.0f}<extra></extra>"))
fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
fig.update_xaxes(title="Profit per period relative to the market average (95% CI)")
st.plotly_chart(style(fig, 380, "Who earns most when every theory is in the same market?"))
st.dataframe(rk.drop(columns="key"), hide_index=True, width="stretch", column_config={
    "agent": "Agent", "theory": "Theory", "mean_rank": st.column_config.NumberColumn("Mean rank", format="%.2f"),
    "win_share": st.column_config.NumberColumn("Wins", format="%.0%"),
    "rel_profit": st.column_config.NumberColumn("Profit vs mean", format="%.0f"),
    "lo": st.column_config.NumberColumn("95% lo", format="%.0f"), "hi": st.column_config.NumberColumn("95% hi", format="%.0f"),
    "profit": st.column_config.NumberColumn("Profit", format="%.0f"),
    "change_rate": st.column_config.NumberColumn("Share of periods output changed", format="%.2f")})
st.markdown("**Reliability-condition agent against each rival** (paired by market, Holm-adjusted)")
st.dataframe(res.pairwise.drop(columns="key"), hide_index=True, width="stretch", column_config={
    "rival": "Rival", "diff": st.column_config.NumberColumn("Profit difference", format="%+.0f"),
    "lo": st.column_config.NumberColumn("95% lo", format="%.0f"), "hi": st.column_config.NumberColumn("95% hi", format="%.0f"),
    "p": st.column_config.NumberColumn("p", format="%.3f"), "holm_reject": "Significant (Holm)"})

st.header("6 · Invasion tests", divider="gray")
st.caption(f"Each cell: one mutant (column) in a market of {pr.residents} residents (row). Value = mutant's profit minus "
           "the residents' mean, relative to the residents' profit. Positive (blue) = the mutant out-earns the "
           "residents and can invade. A type whose row has no significant blue cell resists invasion.")
inv = res.invasion
names = {k: AGENTS[k].name for k in KEYS}
mat = inv.pivot(index="resident", columns="mutant", values="rel_fitness").reindex(index=KEYS, columns=KEYS)
ver = inv.pivot(index="resident", columns="mutant", values="verdict").reindex(index=KEYS, columns=KEYS)
lim = float(np.nanpercentile(np.abs(mat.to_numpy()), 95)) or 1.0
txt = np.where(ver.isna(), "", np.where(ver == "invades", "▲", np.where(ver == "repelled", "▼", "·")))
fh = go.Figure(go.Heatmap(z=mat.to_numpy(), x=[names[k] for k in KEYS], y=[names[k] for k in KEYS], zmin=-lim, zmax=lim,
                          colorscale=DIVERGING, text=txt, texttemplate="%{text}",
                          hovertemplate="residents %{y}<br>mutant %{x}<br>relative fitness %{z:+.2f}<extra></extra>",
                          colorbar=dict(title="mutant vs<br>residents")))
fh.update_yaxes(autorange="reversed", title="Resident population")
fh.update_xaxes(title="Mutant")
st.plotly_chart(style(fh, 480, "▲ invades · ▼ repelled · · no significant difference (Holm-adjusted)"))
resist = [names[k] for k in KEYS if not (ver.loc[k] == "invades").any()]
st.markdown("**Uninvadable populations:** " + (", ".join(resist) if resist else "none: every population can be "
                                                "invaded by at least one rival."))

st.header("7 · Robustness across environments", divider="gray")
st.caption("Global sensitivity analysis: standardised regression coefficients of each agent's profit relative to the "
           "market mean on the environment parameters (all standardised). Blue = the agent does relatively better "
           "as the parameter rises.")
rb = res.robustness
feats = ["delta", "c_max", "q_range", "noise", "foresight", "hazard", "belief_lag"]
flab = ["Volatility Δ", "Max cost", "Quantity range", "Noise σ", "Foresight κ", "Shift hazard λ", "Model lag L"]
fr = go.Figure(go.Heatmap(z=rb[feats].to_numpy(), x=flab, y=rb["agent"], zmin=-1, zmax=1, colorscale=DIVERGING,
                          text=np.round(rb[feats].to_numpy(), 2), texttemplate="%{text}",
                          hovertemplate="%{y}<br>%{x}: %{z:+.2f}<extra></extra>"))
fr.update_yaxes(autorange="reversed")
st.plotly_chart(style(fr, 420, "Where does each rule do relatively well?"))

c1, c2, c3 = st.columns(3)
with c1:
    download(res.h2h, f"arena_head_to_head_{res.prereg_hash}.csv", "Download head-to-head data")
with c2:
    download(res.inv, f"arena_invasion_{res.prereg_hash}.csv", "Download invasion data")
with c3:
    download(res.tuning_log, f"arena_tuning_{res.prereg_hash}.csv", "Download tuning log")

# ------------------------------------------------------------------------------------------------ replication
st.header("8 · Replicate the whole protocol", divider="gray")
st.caption("Tuning and evaluation are repeated with fresh training and test seeds. Conclusions that hold in every "
           "replication are robust; rankings that move between replications are not.")
n_rep = st.number_input("Replications", 1, 10, 3, key="arena_nrep",
                        help="Independent repetitions of tuning and testing (about 35 seconds each for the "
                             "registered plan).")
rkey = (pr.digest, int(n_rep))
reps_store = st.session_state.setdefault("arena_reps", {})
if st.button("Run replications", key="arena_rep_btn"):
    bar = st.progress(0.0, "Starting")
    reps_store[rkey] = replicate(pr, int(n_rep), lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
if rkey in reps_store:
    rp = reps_store[rkey]
    allr = pd.concat([res.ranking.assign(replication=0)[["replication", "agent", "key", "mean_rank"]], rp])
    tab = allr.pivot_table(index="agent", columns="replication", values="mean_rank")
    tab.columns = ["Main run" if c == 0 else f"Rep {c}" for c in tab.columns]
    tab["Range"] = tab.max(1) - tab.min(1)
    st.dataframe(tab.sort_values("Main run").round(2), width="stretch")
    st.caption("Mean rank (1 = best) in each replication. Pre-registered verdicts per replication (PR1–PR4, ✓ = "
               "supported): " + "; ".join(f"rep {r}: {v}" for r, v in rp.drop_duplicates("replication")
                                          [["replication", "verdicts"]].itertuples(index=False)))

st.header("9 · Limits", divider="gray")
st.markdown(
    "* Each theory is represented by one agent design. Another implementation of the same theory could do better or "
    "worse; the designs and their sources are listed above so that critics can propose alternatives.\n"
    "* Agents are compared on profit. Other criteria (survival, stability, regret) can rank them differently.\n"
    "* A frozen plan is only a registration once a third party time-stamps it, for example on OSF.\n"
    "* The market is a stylised cobweb oligopoly; conclusions are about these mechanisms, not about real firms.")
