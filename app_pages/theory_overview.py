import pandas as pd
import streamlit as st

from heiner_abm.registered import TOURNAMENT, TOURNAMENT_PLAN
from heiner_abm.theory_content import THEORIES

st.title("Theories of decision making under uncertainty")
st.markdown(
    "When should a decision maker adapt, and when should it stick to a rule? Eight theories give different answers. "
    "This laboratory implements each of them in the same cobweb oligopoly, where firms commit output before costs and "
    "prices are known, and tests them against one another on equal terms: directional experiments, out-of-sample "
    "forecasts, an agent tournament with equal tuning budgets, and a mechanism study. No theory is the default; each "
    "has its own page with the same sections.")

st.header("1 · The theories at a glance", divider="gray")
st.dataframe(pd.DataFrame([{
    "Theory": f"{t.icon} {t.title}", "Core claim": t.tagline, "Uncertainty is…": t.uncertainty_view,
    "Behaviour changes…": t.when_to_change, "More uncertainty makes flexibility…": t.more_uncertainty}
    for t in THEORIES]), hide_index=True, width="stretch")
cols = st.columns(4)
for k, t in enumerate(THEORIES):
    cols[k % 4].page_link(t.page, label=t.title, icon=t.icon)

st.header("2 · The common testbed", divider="gray")
st.markdown(
    "* **Market.** A homogeneous-product oligopoly with linear demand. A raw-material cost follows a bounded random "
    "walk, and firms decide output before the new cost is known. Demand can shift without warning.\n"
    "* **Information.** Every agent sees the last price and market quantity, its own output and a cost estimate of the "
    "same quality. Model-based agents use the same, possibly outdated, demand model.\n"
    "* **Uncertainty, by source.** Risk (cost volatility), perception error, model misspecification (unannounced "
    "demand shifts and slow model updating), strategic complexity (number of rivals), competence (cost foresight) "
    "and stakes (profitability, fixed costs) can each be varied separately.\n"
    "* **Performance.** Profit, downside risk, survival, volatility, regret and worst case, in held-out "
    "environments.")

st.header("3 · How they fared", divider="gray")
st.caption(f"Agent tournament, registered plan `{TOURNAMENT_PLAN}`: mean rank among nine agents (1 = best) in the "
           "main run and three replications with fresh seeds. Rule B (never change output) is included as a "
           "benchmark.")
rows = []
for key, r in sorted(TOURNAMENT.items(), key=lambda kv: kv[1]["profit"][0]):
    t = next((x for x in THEORIES if x.key == key), None)
    rows.append({"Theory": f"{t.icon} {t.title}" if t else "Rule B (benchmark)", "Design": r["design"],
                 "Profit rank, main": r["profit"][0],
                 "Profit rank, replications": f"{min(r['profit'][1:]):.2f}–{max(r['profit'][1:]):.2f}",
                 "Aggregate rank (six criteria)": f"{min(r['aggregate']):.2f}–{max(r['aggregate']):.2f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"Profit rank, main": st.column_config.NumberColumn(format="%.2f")})
st.markdown(
    "* **Model-free rules lead.** Adaptive price expectations, a target-margin heuristic and an inaction band on a "
    "price-based target are the top three in every run: with unannounced demand shifts, a misspecified model is the "
    "main source of decision error.\n"
    "* **Restriction pays where the flexible rule is unreliable.** Holding back from an error-prone, model-based "
    "target pays when its reliability is known, and breaks even near an error-to-signal ratio of about 1.\n"
    "* **Knowing when to hold back is the hard part.** Agents that must learn their own reliability lose that gain; "
    "fixed rules recover part of it.\n"
    "* **Criteria matter.** Rankings by profit, downside risk and survival differ; most theories are Pareto-efficient.")


def _lit():
    import heiner_abm.literature as lit
    return lit


lit = _lit()
st.header("4 · Hypotheses and the research behind them", divider="gray")
st.dataframe(pd.DataFrame([{
    "ID": h.hid, "Hypothesis": h.title, "Prediction tested": h.rc_prediction,
    "Alternative": f"{h.alt_label}: {h.alt_prediction}",
    "Supporting research": lit.cite(*[k for k, _ in h.support]),
    "Alternative research": lit.cite(*[k for k, _ in h.alternative]), "Where to test": h.where}
    for h in lit.HYPOTHESES]), hide_index=True, width="stretch")
st.caption("Every hypothesis card in the app opens its research basis; the *Research & contribution* page collects "
           "the full bibliography.")

st.header("5 · Contribution to the literature", divider="gray")
for k, (title, text, keys) in enumerate(lit.CONTRIBUTIONS, 1):
    st.markdown(f"**C{k} · {title}.** {text} *({lit.cite(*keys)})*")
