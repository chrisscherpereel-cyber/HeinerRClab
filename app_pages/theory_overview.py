import pandas as pd
import streamlit as st

from heiner_abm.registered import TOURNAMENT, TOURNAMENT_PLAN
from heiner_abm.theory_content import THEORIES

st.title("Theories of decision making under uncertainty")
st.markdown(
    "When should a decision maker adapt, and when should it stick to a rule? Nine theories give different answers. "
    "This laboratory implements each of them in the same cobweb oligopoly, where firms commit output before costs and "
    "prices are known, and tests them against one another on equal terms: directional experiments, out-of-sample "
    "forecasts, an agent tournament with equal tuning budgets, a mechanism study, endogenous rule choice, validation "
    "against field patterns and laboratory data, three further decision tasks and a benchmark solvable on paper. No "
    "theory is privileged: choose the one to test in the sidebar, and each has its own page with the same sections.")

st.info(f"**Theory under test:** ⭐ {next(t.title for t in THEORIES if t.key == st.session_state.get('focal_theory', 'heiner'))}. "
        "Choose any of the nine in the sidebar; its predictions are then highlighted on every hypothesis card, in "
        "the research tables and in the tournament reviews.", icon="🎯")
st.header("1 · The theories at a glance", divider="gray")
st.dataframe(pd.DataFrame([{
    "Theory": ("⭐ " if t.key == st.session_state.get("focal_theory", "heiner") else "") + f"{t.icon} {t.title}", "Core claim": t.tagline, "Uncertainty is…": t.uncertainty_view,
    "Behavior changes…": t.when_to_change, "More uncertainty makes flexibility…": t.more_uncertainty}
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
st.caption(f"Agent tournament, registered plan `{TOURNAMENT_PLAN}`: mean rank among ten agents (1 = best) in the "
           "main run and three replications with fresh seeds. Rule B (never change output) is included as a "
           "benchmark.")
rows = []
for key, r in sorted(TOURNAMENT.items(), key=lambda kv: kv[1]["profit"][0]):
    t = next((x for x in THEORIES if x.key == key), None)
    star = "⭐ " if t and t.key == st.session_state.get("focal_theory", "heiner") else ""
    rows.append({"Theory": f"{star}{t.icon} {t.title}" if t else "Rule B (benchmark)", "Design": r["design"],
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
    "* **Criteria matter.** Rankings by profit, downside risk and survival differ; most theories are Pareto-efficient.\n"
    "* **Inertia is rewarded on the downside.** Organizational ecology's crisis-driven agent is mid-field on profit "
    "but has the best aggregate rank over six criteria in the main run, through low volatility and high survival.\n"
    "* **When firms choose their own rules,** populations shift toward restricted rules and abandon the flexible "
    "optimizer as uncertainty rises, but they do not change output less often, and selection favors a simple "
    "target-margin heuristic as much as an explicit restriction (Rule choice).\n"
    "* **The boundary generalizes.** In an inventory task, a learning task with shifting payoffs and an irreversible "
    "investment task, restriction pays only where the flexible rule is unreliable, with a break-even error-to-signal "
    "ratio near or below 1 (Generalization).\n"
    "* **What is uniquely Heiner's.** In a single-firm benchmark the simulation reproduces the exact Muth–Kalman "
    "optimum; with lopsided stakes, restricting the filter's costly moves cuts the loss by up to a third although the "
    "information is unchanged, as the reliability condition predicts and certainty equivalence does not. A filter "
    "that builds the stakes into its estimate does better still (Solvable benchmark).\n"
    "* **Validation.** The market reproduces four of six documented field patterns (not lumpy adjustment or excess volatility). In five "
    "public experimental datasets, behavior is heterogeneous, simulated Cournot markets reach the human level of "
    "competition and time pressure shifts people toward simple or restricted rules; but the learned reliability "
    "condition does not predict individual choices better than the flexible rules it restricts, and people change "
    "their decisions less often than any fitted rule implies (Empirical validation).")


def _lit():
    import heiner_abm.literature as lit
    return lit


lit = _lit()
st.header("4 · Hypotheses and the research behind them", divider="gray")
from heiner_abm.focal import prediction  # noqa: E402
from ui.common import focal_theory, focal_title  # noqa: E402
st.caption(f"The first prediction column is the theory under test, ⭐ {focal_title()}; change it in the sidebar.")
st.dataframe(pd.DataFrame([{
    "ID": h.hid, "Hypothesis": h.title, f"⭐ {focal_title()} predicts": prediction(h.hid, focal_theory())[0],
    "Reliability condition predicts": h.rc_prediction,
    "Alternative": f"{h.alt_label}: {h.alt_prediction}",
    "Supporting research": lit.cite(*[k for k, _ in h.support]),
    "Alternative research": lit.cite(*[k for k, _ in h.alternative]), "Where to test": h.where}
    for h in lit.HYPOTHESES]), hide_index=True, width="stretch")
st.caption("Every hypothesis card in the app opens its research basis; the *Research & contribution* page collects "
           "the full bibliography.")

st.header("5 · Contribution to the literature", divider="gray")
for k, (title, text, keys) in enumerate(lit.CONTRIBUTIONS, 1):
    st.markdown(f"**C{k} · {title}.** {text} *({lit.cite(*keys)})*")
