from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.arena import (DESIGNS, FEATURES, KEYS, PREREG, QUICK, THEORY_DESIGNS, THEORY_NAMES, Prereg,
                              design_budget,
                              pairwise, replicate, run_protocol)
from heiner_abm.literature import cite
from ui.common import CAT, DIVERGING, download, focal_theory, focal_title, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note, set_engine

st.title("Agent tournament: rival theories as competing agents")
evidence_note("simulation")
set_engine("arena")
st.caption("Each theory of flexibility under uncertainty is implemented as decision rules. The rules compete in the "
           "same cobweb market, with the same information and the same random shocks, under a fairness protocol "
           "fixed in advance. This replaces predictions written by the modeler with behavior that can win or lose.")

hypothesis_card(
    "ARENA",
    "Every theory is represented by **two designs**, and enters the tournament with whichever design did better on "
    "the training environments. Most designs combine a **target** (a model-based best reply, or a price-based "
    "adaptive expectation that needs no demand model) with a **selection rule** that decides when to move: always "
    "(optimization, cobweb theory), outside an inaction band (an inaction-band heuristic inspired by real options, "
    "not a dynamic real-options model), when the learned reliability condition "
    "holds (Heiner), or when profit falls below aspiration (satisficing). Because the targets are shared, the "
    "selection-rule experiment isolates the role of the selection rule: which way of deciding *when* to move works "
    "best toward the same target?",
    notes={"heiner": "Deciding when to deviate by its reliability should beat always adjusting toward the same "
                     "target where the target is error-prone."})

st.header("1 · The fairness protocol", divider="gray")
st.markdown(
    "1. **Same market, same information.** Every agent sees last period's price and market quantity, its own output "
    "and a cost estimate of the same quality. Model-based agents use the same (possibly outdated) demand model.\n"
    "2. **Two designs per theory, equal budget per design.** Every design's free parameters are tuned with the same "
    "number of candidate settings (Latin-hypercube search, defaults included) on *training* environments, in two "
    "rounds. Each theory then enters with whichever design scored higher on training data, never on test data. "
    "Budgets are equal in **search density**, not in raw evaluations: designs carry between one and six free "
    "parameters, so each gets the same number of candidates *per free parameter* and none is penalized for a "
    "richer parameterization (see section 4).\n"
    "3. **Held-out, randomly drawn environments.** Test environments come from pre-registered ranges with different "
    "seeds and are never used for tuning or design selection.\n"
    "4. **Several criteria and yardsticks.** Mean profit, downside risk (CVaR 5%), survival against a capital buffer, "
    "volatility, regret and worst case; a target × selection-rule experiment; invasion tests; global sensitivity "
    "analysis; and replication of the whole protocol with fresh seeds.\n"
    "5. **A frozen plan.** Ranges, budgets, hypotheses, decision rules and the agent and analysis code are hashed. A "
    "run with any change is labeled exploratory.")
st.caption("Methods: " + cite("axelrod1984", "maynardsmith1973", "mckay1979", "bergstra2012", "holm1979",
                               "rockafellar2000", "savage1951", "saltelli2008", "nosek2018"))

st.subheader("The designs")
st.dataframe(pd.DataFrame([dict(Theory=THEORY_NAMES[t], Design=DESIGNS[d].name, Rule=DESIGNS[d].rule,
                                Parameters=", ".join(f"{n} ∈ [{s.lo:g}, {s.hi:g}]" for n, s in DESIGNS[d].SPACE.items())
                                or "–", Sources=cite(*DESIGNS[d].sources))
                           for t, ds in THEORY_DESIGNS.items() for d in ds]),
             hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ plan
st.header("2 · Pre-registered plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Pre-registered", "Quick check", "Custom"], horizontal=True, key="arena_scale",
                 help="Pre-registered runs the frozen plan (about 1.5 minutes). Quick check is a small exploratory "
                      "run for trying things out. Custom lets you change the settings; the run is then exploratory.")
pr = PREREG if scale == "Pre-registered" else QUICK
if scale == "Custom":
    c = st.columns(4)
    budget = c[0].number_input("Tuning candidates per free parameter", 1, 100, PREREG.budget_per_parameter,
                               help="Candidate settings per free parameter, so every design's space is searched at "
                                    "the same density. A design with six free parameters therefore gets six times "
                                    "the candidates of a one-parameter design.")
    n_train = c[1].number_input("Training environments", 4, 200, PREREG.n_train,
                                help="Environments used only for tuning and design selection.")
    n_test = c[2].number_input("Test environments", 4, 300, PREREG.n_test,
                               help="Held-out environments used for every result.")
    periods = c[3].number_input("Periods per market", 150, 5000, PREREG.periods, step=50,
                                help="Length of every simulated market.")
    c = st.columns(4)
    rounds = c[0].number_input("Tuning rounds", 1, 4, PREREG.rounds,
                               help="Round r tunes each design against the others' round r − 1 settings.")
    residents = c[1].number_input("Residents in invasion tests", 2, 10, PREREG.residents,
                                  help="Firms of the resident type; one mutant joins them.")
    train_seed = c[2].number_input("Training seed", 0, 10**6, PREREG.train_seed, help="Seed for training environments.")
    test_seed = c[3].number_input("Test seed", 0, 10**6, PREREG.test_seed, help="Seed for test environments.")
    pr = Prereg(**{**asdict(PREREG), "budget_per_parameter": int(budget), "n_train": int(n_train), "n_test": int(n_test),
                   "periods": int(periods), "rounds": int(rounds), "residents": int(residents),
                   "train_seed": int(train_seed), "test_seed": int(test_seed)})
_budgets = [design_budget(pr, d) for d in DESIGNS]
min_k, max_k = min(_budgets), max(_budgets)
exploratory = pr.digest != PREREG.digest
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{pr.digest}` " + ("· **exploratory** (differs from the registered plan "
                                                     f"`{PREREG.digest}`)" if exploratory else "· registered plan"))
    st.caption(f"{pr.n_train} training and {pr.n_test} × {pr.reps_test} test environments · {pr.periods} periods "
               f"(burn-in {pr.burn_in}) · {pr.budget_per_parameter} tuning candidates per free parameter "
               f"({min_k}–{max_k} per design) × 2 designs per theory × {pr.rounds} rounds · "
               f"invasion: {pr.residents} residents + 1 mutant · α = {pr.alpha}")
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

# ------------------------------------------------------------------------------------------------ verdicts
st.header("3 · Pre-registered hypotheses", divider="gray")
if res.exploratory:
    st.warning("Exploratory run: the plan differs from the registered one, so these verdicts are not confirmatory.",
               icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject",
            f"**{r['id']}** · {r['hypothesis']} **{r['verdict'].capitalize()}**: {r['result']}.")

# ------------------------------------------------------------------------------------------------ designs
st.header("4 · Design selection and tuning", divider="gray")
log = res.tuning_log
last = log[log["round"] == log["round"].max()].copy()
last["theory"] = last["theory"].map(THEORY_NAMES)
if "n_params" not in last:                      # older cached runs
    last["n_params"] = last["design"].map(lambda d: len(DESIGNS[d].SPACE))
    last["n_candidates"] = pr.budget
    last["candidates_per_param"] = pr.budget / last["n_params"].clip(lower=1)
last = last.rename(columns={"candidates_per_param": "per_param"})
st.dataframe(last[["theory", "design_name", "selected", "n_params", "n_candidates", "per_param", "default_score",
                   "best_score", "gain"] + [c for c in last if c.startswith("param:")]],
             hide_index=True, width="stretch", column_config={
                 "theory": "Theory", "design_name": "Design", "selected": "Enters the tournament",
                 "n_params": st.column_config.NumberColumn("Free parameters", format="%d"),
                 "n_candidates": st.column_config.NumberColumn("Candidates evaluated", format="%d"),
                 "per_param": st.column_config.NumberColumn("Candidates per parameter", format="%.1f"),
                 "default_score": st.column_config.NumberColumn("Profit, default settings", format="%.0f"),
                 "best_score": st.column_config.NumberColumn("Profit, tuned", format="%.0f"),
                 "gain": st.column_config.NumberColumn("Gain from tuning", format="%+.0f")})
st.caption(f"Final tuning round, training environments only. Every design was searched at the same density, "
           f"{pr.budget_per_parameter} candidate settings per free parameter ({min_k}–{max_k} candidates depending on "
           "the design); each theory enters with its higher-scoring design. Until 7 October 2026 every design got the "
           "same *number* of candidates whatever its dimension, which searched the six-parameter "
           "reliability-condition design six times less thoroughly than a one-parameter design and so handicapped the "
           "focal theory; `Prereg.budget_rule = \"per_design\"` restores that behavior.")

# ------------------------------------------------------------------------------------------------ head to head
st.header("5 · Head-to-head on several criteria", divider="gray")
rk = res.ranking
fig = go.Figure(go.Bar(
    y=rk["agent"][::-1], x=rk["rel_profit"][::-1], orientation="h",
    marker_color=[CAT[0] if k == focal_theory() else "#9aa0a6" for k in rk["key"][::-1]],
    error_x=dict(type="data", symmetric=False, array=(rk["hi"] - rk["rel_profit"])[::-1],
                 arrayminus=(rk["rel_profit"] - rk["lo"])[::-1], thickness=1.2, width=4),
    customdata=rk["design_name"][::-1], hovertemplate="%{y}<br>%{customdata}<br>profit vs market mean %{x:,.0f}"
                                                      "<extra></extra>"))
fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
fig.update_xaxes(title="Profit per period relative to the market average (95% CI)")
st.plotly_chart(style(fig, 380, "Mean profit in held-out mixed markets"))
cols = ["agent", "design_name", "mean_rank", "rel_profit", "cvar5", "survived", "sd", "regret", "worst10",
        "aggregate_rank", "pareto", "change_rate"]
st.dataframe(rk[cols].sort_values("aggregate_rank"), hide_index=True, width="stretch", column_config={
    "agent": "Theory", "design_name": "Design", "mean_rank": st.column_config.NumberColumn("Profit rank", format="%.2f"),
    "rel_profit": st.column_config.NumberColumn("Profit vs mean", format="%.0f"),
    "cvar5": st.column_config.NumberColumn("CVaR 5%", format="%.0f",
                                           help="Mean profit in the worst 5% of periods (higher is better)."),
    "survived": st.column_config.NumberColumn("Survival", format="%.0%",
                                              help="Share of markets in which cumulative profit never fell below "
                                                   "−20 periods of Nash profit."),
    "sd": st.column_config.NumberColumn("Volatility", format="%.0f", help="S.d. of per-period profit (lower is better)."),
    "regret": st.column_config.NumberColumn("Regret", format="%.0f",
                                            help="Shortfall from the best firm in the same market (lower is better)."),
    "worst10": st.column_config.NumberColumn("Worst case", format="%.0f",
                                             help="10th percentile of profit vs mean across markets (higher is better)."),
    "aggregate_rank": st.column_config.NumberColumn("Aggregate rank", format="%.2f",
                                                    help="Mean of the ranks on the six criteria."),
    "pareto": st.column_config.CheckboxColumn("Pareto-efficient",
                                              help="No other agent is at least as good on every criterion and "
                                                   "better on one."),
    "change_rate": st.column_config.NumberColumn("Output changed", format="%.2f",
                                                 help="Share of periods in which output changed (predictability).")})
st.caption("Sorted by aggregate rank across the six criteria. " + cite("rockafellar2000", "savage1951", "alchian1950"))
_focal = focal_theory()
if _focal == "heiner":
    pw = res.pairwise
    st.markdown("**⭐ Reliability-condition agent against each rival** (profit, paired by market, Holm-adjusted; "
                "pre-registered as PR1)")
else:
    pw = pairwise(res.h2h, _focal, pr)
    st.markdown(f"**⭐ {focal_title()} agent against each rival** (profit, paired by market, Holm-adjusted). "
                "The pre-registered head-to-head (PR1) names the reliability-condition agent, so this comparison for "
                "the theory chosen in the sidebar is exploratory.")
st.dataframe(pw.drop(columns="key"), hide_index=True, width="stretch", column_config={
    "rival": "Rival", "diff": st.column_config.NumberColumn("Profit difference", format="%+.0f"),
    "lo": st.column_config.NumberColumn("95% lo", format="%.0f"), "hi": st.column_config.NumberColumn("95% hi", format="%.0f"),
    "p": st.column_config.NumberColumn("p", format="%.3f"), "holm_reject": "Significant (Holm)"})

# ------------------------------------------------------------------------------------------------ selection rules
st.header("6 · Selection-rule experiment: the same targets, four ways of deciding when to move", divider="gray")
st.caption("Eight composite designs compete in the same held-out markets: two targets × four selection rules, each "
           "with its own tuned parameters. Each cell compares a selection rule with 'always' on the same target "
           "(paired by market).")
se = res.selection
for crit, title in (("profit", "Profit difference vs always"), ("cvar5", "Downside (CVaR 5%) difference vs always")):
    d = se[se["criterion"] == crit]
    z = d.pivot(index="select", columns="target", values="diff").reindex(["rc", "band", "aspiration"])
    txt = d.assign(t=lambda x: x.apply(lambda r: f"{r['diff']:+.0f}<br>[{r['lo']:.0f}, {r['hi']:.0f}]", axis=1)) \
        .pivot(index="select", columns="target", values="t").reindex(z.index)
    lim = float(np.nanmax(np.abs(z.to_numpy()))) or 1.0
    fz = go.Figure(go.Heatmap(z=z.to_numpy(), x=["Model-based target", "Price-based target"],
                              y=["Reliability condition (Heiner)", "Inaction-band heuristic (real-options inspired)",
                                 "Aspiration (satisficing)"], zmin=-lim, zmax=lim, colorscale=DIVERGING,
                              text=txt.to_numpy(), texttemplate="%{text}", hoverinfo="skip"))
    fz.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fz, 300, title + " (95% CI; blue = better than always)"))
st.dataframe(res.factorial[["agent", "mean_rank", "rel_profit", "cvar5", "survived", "aggregate_rank", "pareto"]],
             hide_index=True, width="stretch", column_config={
                 "agent": "Design", "mean_rank": st.column_config.NumberColumn("Profit rank", format="%.2f"),
                 "rel_profit": st.column_config.NumberColumn("Profit vs mean", format="%.0f"),
                 "cvar5": st.column_config.NumberColumn("CVaR 5%", format="%.0f"),
                 "survived": st.column_config.NumberColumn("Survival", format="%.0%"),
                 "aggregate_rank": st.column_config.NumberColumn("Aggregate rank", format="%.2f"),
                 "pareto": st.column_config.CheckboxColumn("Pareto-efficient")})

# ------------------------------------------------------------------------------------------------ invasion
st.header("7 · Invasion tests", divider="gray")
st.caption(f"Each cell: one mutant (column) in a market of {pr.residents} residents (row), each theory with its "
           "selected design. Value = mutant's profit minus the residents' mean, relative to the residents' profit. "
           "Positive (blue) = the mutant out-earns the residents and can invade.")
inv = res.invasion
mat = inv.pivot(index="resident", columns="mutant", values="rel_fitness").reindex(index=KEYS, columns=KEYS)
ver = inv.pivot(index="resident", columns="mutant", values="verdict").reindex(index=KEYS, columns=KEYS)
lim = float(np.nanpercentile(np.abs(mat.to_numpy()), 95)) or 1.0
txt = np.where(ver.isna(), "", np.where(ver == "invades", "▲", np.where(ver == "repelled", "▼", "·")))
names = [THEORY_NAMES[k] for k in KEYS]
fh = go.Figure(go.Heatmap(z=mat.to_numpy(), x=names, y=names, zmin=-lim, zmax=lim, colorscale=DIVERGING, text=txt,
                          texttemplate="%{text}",
                          hovertemplate="residents %{y}<br>mutant %{x}<br>relative fitness %{z:+.2f}<extra></extra>",
                          colorbar=dict(title="mutant vs<br>residents")))
fh.update_yaxes(autorange="reversed", title="Resident population")
fh.update_xaxes(title="Mutant")
st.plotly_chart(style(fh, 480, "▲ invades · ▼ repelled · · no significant difference (Holm-adjusted)"))
resist = [THEORY_NAMES[k] for k in KEYS if not (ver.loc[k] == "invades").any()]
st.markdown("**Uninvadable populations:** " + (", ".join(resist) if resist else "none: every population can be "
                                                "invaded by at least one rival."))

# ------------------------------------------------------------------------------------------------ robustness
st.header("8 · Robustness across environments", divider="gray")
st.caption("Global sensitivity analysis: standardized regression coefficients of each agent's profit relative to the "
           "market mean on the environment parameters. Blue = the agent does relatively better as the parameter "
           "rises.")
rb = res.robustness
flab = ["Volatility Δ", "Max cost", "Quantity range", "Noise σ", "Foresight κ", "Shift hazard λ", "Model lag L"]
fr = go.Figure(go.Heatmap(z=rb[FEATURES].to_numpy(), x=flab, y=rb["agent"], zmin=-1, zmax=1, colorscale=DIVERGING,
                          text=np.round(rb[FEATURES].to_numpy(), 2), texttemplate="%{text}",
                          hovertemplate="%{y}<br>%{x}: %{z:+.2f}<extra></extra>"))
fr.update_yaxes(autorange="reversed")
st.plotly_chart(style(fr, 420, "Where does each rule do relatively well?"))

c1, c2, c3, c4 = st.columns(4)
with c1:
    download(res.h2h, f"arena_head_to_head_{res.prereg_hash}.csv", "Head-to-head data")
with c2:
    download(res.fac, f"arena_selection_rules_{res.prereg_hash}.csv", "Selection-rule data")
with c3:
    download(res.inv, f"arena_invasion_{res.prereg_hash}.csv", "Invasion data")
with c4:
    download(res.tuning_log, f"arena_tuning_{res.prereg_hash}.csv", "Tuning log")

# ------------------------------------------------------------------------------------------------ replication
st.header("9 · Replicate the whole protocol", divider="gray")
st.caption("Tuning, design selection and evaluation are repeated with fresh training and test seeds. Conclusions that "
           "hold in every replication are robust; rankings that move between replications are not.")
n_rep = st.number_input("Replications", 1, 10, 3, key="arena_nrep",
                        help="Independent repetitions (about 1.5 minutes each for the registered plan).")
rkey = (pr.digest, int(n_rep))
reps_store = st.session_state.setdefault("arena_reps", {})
if st.button("Run replications", key="arena_rep_btn"):
    bar = st.progress(0.0, "Starting")
    reps_store[rkey] = replicate(pr, int(n_rep), lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
if rkey in reps_store:
    rp = reps_store[rkey]
    main = res.ranking.assign(replication=0)[["replication", "agent", "mean_rank", "aggregate_rank"]]
    allr = pd.concat([main, rp[["replication", "agent", "mean_rank", "aggregate_rank"]]])
    for col, title in (("mean_rank", "Profit rank"), ("aggregate_rank", "Aggregate rank (six criteria)")):
        tab = allr.pivot_table(index="agent", columns="replication", values=col)
        tab.columns = ["Main run" if c == 0 else f"Rep {c}" for c in tab.columns]
        tab["Range"] = tab.max(1) - tab.min(1)
        st.markdown(f"**{title}** (1 = best)")
        st.dataframe(tab.sort_values("Main run").round(2), width="stretch")
    sel = rp.attrs.get("selection")
    if sel is not None and len(sel):
        allsel = pd.concat([res.selection.assign(replication=0), sel])
        allsel["cell"] = allsel.apply(lambda r: f"{r['diff']:+.0f}" + ("*" if r["lo"] > 0 or r["hi"] < 0 else ""), axis=1)
        tab = allsel.pivot_table(index=["criterion", "target", "select"], columns="replication", values="cell",
                                 aggfunc="first")
        tab.columns = ["Main run" if c == 0 else f"Rep {c}" for c in tab.columns]
        st.markdown("**Selection rule vs always, same target** (* = 95% CI excludes 0)")
        st.dataframe(tab, width="stretch")
    st.caption("Designs selected per replication and pre-registered verdicts (PR1–PR6, ✓ = supported): "
               + "; ".join(f"rep {r}: {v}" for r, v in rp.drop_duplicates("replication")
                           [["replication", "verdicts"]].itertuples(index=False)))

st.header("10 · Limits", divider="gray")
st.markdown(
    "* Each theory is represented by two designs. Other implementations could do better or worse; the designs and "
    "their sources are listed above so that critics can propose alternatives, and new designs plug into the same "
    "protocol.\n"
    "* Survival is measured against a fixed capital buffer without removing ruined firms, so market composition stays "
    "comparable across agents.\n"
    "* A frozen plan is only a registration once a third party time-stamps it, for example on OSF.\n"
    "* The market is a stylized cobweb oligopoly; conclusions are about these mechanisms, not about real firms.")
