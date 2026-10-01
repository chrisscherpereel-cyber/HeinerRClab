"""The theory under test: what each of the eight theories predicts for each hypothesis.

The user picks a focal theory; hypothesis cards, overview tables and the review sections then highlight that theory's
prediction and show the others as competitors. Heiner's reliability condition is one choice among eight.

A theory's prediction for a hypothesis is taken, in this order of precedence, from
    1. the registry itself: the reliability-condition prediction (Heiner) or the registered alternative, when the
       alternative belongs to the theory;
    2. the directional tournament (heiner_abm.theories.EXPERIMENTS), whose experiments state each theory's prediction
       and the reason for it;
    3. the statements below, written from the theory's core claim for hypotheses the first two sources do not cover;
    4. otherwise the theory's general stance (what triggers change, and what more uncertainty does to flexibility),
       labelled as such.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .theory_content import THEORIES, THEORY_BY_KEY

FOCAL_KEYS: List[str] = [t.key for t in THEORIES]
DEFAULT_FOCAL = "heiner"

# which theories the registered alternative of a hypothesis speaks for (matched on its label)
_ALT_PATTERNS = (
    (r"optimis|marginal|rational expectations|value of information|optimal|frictionless", "optimiser"),
    (r"real options|preference for flexibility|\(s, s\)", "options"),
    (r"cobweb|adaptive expectations|best.reply dynamics", "cobweb"),
    (r"satisficing|behavioural theory", "satisficing"),
    (r"reinforcement|bayesian learning|adaptive learning", "rl"),
    (r"evolutionary|ecology|imitation", "imitation"),
)
_SIGN_WORDS = {"+": "a positive effect", "-": "a negative effect", "0": "no effect", ">=0": "no harm (non-negative)"}


def alternative_owners(alt_label: str) -> List[str]:
    lab = alt_label.lower()
    return [key for pat, key in _ALT_PATTERNS if re.search(pat, lab)]


# statements written from each theory's core claim (step 3 above)
STATEMENTS: Dict[str, Dict[str, str]] = {
    "optimiser": {
        "EMPVAL": "Best replies and rational expectations predict human choices best out of sample.",
        "H2": "Flexibility pays at every profit level: an optimiser uses it only when it raises expected profit.",
        "H5": "Competition changes margins, not the value of responding optimally to new information.",
        "H6": "Markets converge to the Cournot–Nash equilibrium; rigidity only delays convergence.",
        "H8": "An optimiser filters noise optimally, so noise lowers the value of responding but never makes it negative.",
        "H9": "The rule that moves to the best available target every period does best.",
        "H11": "More rivals change the target, not the value of best-responding to it.",
        "H12": "A correctly specified optimiser updates its model at once; lags are a misspecification, not a feature.",
        "SDT": "Decisions follow a likelihood-ratio criterion; no separate restriction to rules is needed.",
        "KNIGHT": "Structural uncertainty is one more risk to average over; it does not change the case for optimising.",
        "PUNCT": "Adjustment is smooth and proportional to the size of the shock.",
        "CDGAP": "Competence raises the value of flexibility; difficulty lowers it but never below zero.",
        "ARENA": "The optimising design earns at least as much as every rival in held-out markets.",
        "MECH2": "No kind of uncertainty makes restriction pay for a correctly specified optimiser.",
        "PATTERN": "Under rational expectations there are no systematic cycles or stickiness; such patterns reflect "
                   "frictions.",
        "CALIB": "Choices are best predicted by rational expectations or best replies; deviations are noise.",
        "EXPER": "People who respond to new information optimally do best; more uncertainty does not make restraint pay.",
        "GEN": "Optimal policies use all available information; restricting a well-specified flexible rule never pays.",
    },
    "options": {
        "EMPVAL": "Inaction bands predict human choices best: people adjust only when the gap is large.",
        "H1": "An option is never worth less than zero: free flexibility cannot hurt.",
        "H2": "The value of flexibility depends on uncertainty and adjustment costs, not on the level of profit.",
        "H5": "Competition erodes option value only through the payoffs from exercising the option.",
        "H6": "Inaction bands make prices sticky, but markets track equilibrium on average.",
        "H9": "Rules that move only when the gap to the target exceeds a band win, and the band widens with "
              "uncertainty.",
        "H11": "More rivals do not change the option logic directly.",
        "H12": "Slower learning raises uncertainty, which raises the value of waiting (a wider band).",
        "RC": "Which firms gain from flexibility is predicted by option value: the more volatility, the more value.",
        "DRC": "Decisions are valued as options over the whole horizon, including the value of waiting.",
        "SDT": "Act when the gap exceeds a threshold set by volatility and adjustment costs.",
        "PUNCT": "Adjustment is lumpy: long inaction, then a large move when the gap crosses the band.",
        "CDGAP": "More uncertainty raises option value; competence lowers uncertainty and with it the option's value.",
        "ARENA": "The inaction-band design does best where volatility is high.",
        "MECH1": "An inaction band captures the gain from not chasing noise without estimating any reliability.",
        "MECH2": "Volatility (risk) raises the value of waiting more than anything else.",
        "EMERGE": "Firms drift toward band rules as volatility rises, because waiting gains value.",
        "PATTERN": "Inaction bands produce sticky, lumpy adjustment ((S, s) behaviour).",
        "CALIB": "Subjects adjust only when the gap to their target is large.",
        "EXPER": "People wait more under high uncertainty because waiting has option value, and waiting pays.",
        "GEN": "Inaction pays where volatility is high relative to adjustment costs; without adjustment costs a band "
               "has no role.",
        "TRACK": "Without adjustment costs there is no value of waiting; stakes shift the level of the action, not its "
                 "timing.",
    },
    "cobweb": {
        "EMPVAL": "Adaptive expectations and demand chasing predict human choices best.",
        "H1": "Free flexibility hurts only by destabilising the market; in a stable market it does not.",
        "H2": "Profitability does not affect stability; only adjustment speed relative to the stability limit does.",
        "H4": "Fixed costs do not change the adjustment dynamics.",
        "H5": "Steeper competitive responses shrink the stability region, so fast adjusters suffer.",
        "H9": "Slowly adjusting rules win when fast adjustment would make the market oscillate.",
        "H10": "Adaptive firms adjust every period whatever the uncertainty; only the size of changes varies.",
        "H12": "Learning lags matter only through their effect on stability.",
        "RC": "Who gains from flexibility is predicted by stability: fast adjusters lose in unstable markets.",
        "DRC": "A decision's consequences propagate through the market's dynamics; stability analysis captures them.",
        "KNIGHT": "Only adjustment dynamics matter, not the source of uncertainty.",
        "PUNCT": "Adjustment is continuous and geometric, not punctuated.",
        "CDGAP": "Difficulty and competence matter only through the stability of adjustment.",
        "ARENA": "Adaptive price expectations, needing no demand model, do well whenever the market is stable.",
        "MECH1": "Restriction pays only by slowing adjustment enough to stabilise the market.",
        "EMERGE": "Selection favours slow adjusters in unstable markets and fast ones in stable markets.",
        "PATTERN": "Naive expectations produce cobweb cycles; adaptive expectations damp them.",
        "CALIB": "Subjects' forecasts follow adaptive expectations.",
        "EXPER": "People adjust by a fraction of the latest error every period; uncertainty does not change how often "
                 "they move.",
        "GEN": "Outside markets there is no feedback from actions to outcomes, so stability theory makes no prediction.",
        "TRACK": "With a single firm there is no market feedback; the theory makes no prediction about stakes.",
    },
    "heuristic": {
        "EMPVAL": "Simple heuristics (anchoring, trend following) predict best, and people differ in which.",
        "H2": "Accuracy, not profitability, decides between simple and flexible rules.",
        "H9": "Simple rules that ignore information win when estimation noise is high.",
        "H10": "Simple rules make behaviour predictable, and their use rises with noise.",
        "H12": "Slower learning raises estimation error, which favours simple rules.",
        "RC": "Who gains from flexibility is predicted by accuracy (estimation error) alone; stakes do not matter.",
        "SDT": "Only discrimination accuracy matters; how the decision criterion depends on stakes is not part of the "
               "theory.",
        "KNIGHT": "Any source of estimation error favours simple rules.",
        "CDGAP": "The gap matters only through estimation error (variance).",
        "ARENA": "A simple target-margin heuristic does as well as complex rules out of sample.",
        "MECH1": "Simple fixed rules capture most of the gain without estimating reliability.",
        "MECH2": "Restriction pays as estimation error rises, whatever its source.",
        "EMERGE": "Simple heuristics spread under uncertainty because they are robust to noise.",
        "CALIB": "Subjects use simple heuristics (anchoring, trend following), and people differ in which.",
        "EXPER": "People rely more on simple rules under uncertainty, and those who do earn more.",
        "GEN": "Restriction pays where estimation error is high, in any task.",
        "TRACK": "Only accuracy matters, so lopsided stakes should not change which rule is best.",
    },
    "satisficing": {
        "EMPVAL": "People change after disappointing outcomes; aspiration rules predict their changes.",
        "H1": "Flexibility as such is not the issue; poor performance triggers search and change.",
        "H3": "Volatility pushes results below aspiration more often, triggering more search and change.",
        "H9": "Rules that change only when performance falls below aspiration do well.",
        "RC": "Who gains from flexibility is predicted by performance relative to aspiration.",
        "CDGAP": "Harder environments lower performance relative to aspiration and raise search.",
        "ARENA": "The aspiration-based design is robust but not the most profitable.",
        "EMERGE": "Under uncertainty more firms fall below aspiration and search, so behaviour becomes less "
                  "predictable.",
        "PATTERN": "Change comes in bursts after poor results.",
        "CALIB": "Subjects update only after a large error or a disappointing result.",
        "EXPER": "People change output more often under high uncertainty because outcomes fall below aspiration more "
                 "often.",
        "GEN": "Change is triggered by poor recent outcomes, not by the reliability of the flexible rule.",
    },
    "rl": {
        "H1": "Flexibility costs a learner while it explores; with enough experience, learned values make good use of "
              "it.",
        "H3": "Volatility makes learned values outdated, so learning becomes noisier and exploration more costly.",
        "H9": "Rules that learn action values from experience adapt to whatever works; no fixed restriction is "
              "needed.",
        "H10": "Higher uncertainty makes value estimates noisier, so choices become less predictable (more "
               "exploration).",
        "RC": "Who gains from flexibility is predicted by learned value differences.",
        "ARENA": "A value learner finds a good response through experience, given enough data.",
        "MECH1": "Learned estimates converge to true values with enough experience, so learned and oracle rules "
                 "coincide.",
        "EMERGE": "Selection on recent payoffs is itself reinforcement: rules that paid recently spread.",
        "CALIB": "Subjects' choices follow reinforcement of past payoffs.",
        "GEN": "A learner that tracks volatility adapts its learning rate, so restriction is unnecessary.",
        "TRACK": "A learner that optimises the true loss learns the stakes; restriction becomes unnecessary.",
    },
    "imitation": {
        "EMPVAL": "Where people see rivals' outcomes, imitating the most successful predicts their choices.",
        "H10": "Selection favours inert, reliable organisations regardless of uncertainty.",
        "ARENA": "Imitating successful rivals performs as well as the rules being imitated.",
        "PATTERN": "Imitating the most profitable firm pushes output above the Cournot–Nash level.",
        "CALIB": "Subjects imitate the most successful group member when they see others' profits.",
        "EXPER": "Participants copy what worked for others; uncertainty blurs who is successful.",
        "GEN": "Without others to imitate, the theory makes no prediction.",
        "TRACK": "With a single firm there is no one to imitate; the theory makes no prediction.",
    },
}


def _lit():
    import heiner_abm.literature as lit
    return lit


def prediction(hid: str, theory: str) -> Tuple[str, str]:
    """(prediction, source) for a theory and a hypothesis; source is 'registry', 'tournament', 'theory' or
    'general'."""
    lit = _lit()
    h = lit.HYPOTHESIS_BY_ID.get(hid)
    t = THEORY_BY_KEY[theory]
    if h is not None:
        if theory == "heiner":
            return h.rc_prediction, "registry"
        if theory in alternative_owners(h.alt_label):
            return h.alt_prediction, "registry"
    if t.tournament_key:
        from .theories import EXPERIMENTS
        for e in EXPERIMENTS:
            if e.hid == hid and e.why.get(t.tournament_key):
                sign = e.predictions.get(t.tournament_key)
                head = f"Predicts {_SIGN_WORDS.get(sign, 'an effect')} ({e.statistic[0].lower() + e.statistic[1:]}). " if sign else ""
                return head + e.why[t.tournament_key], "tournament"
    if hid in STATEMENTS.get(theory, {}):
        return STATEMENTS[theory][hid], "theory"
    return (f"No prediction specific to this hypothesis. General stance: behaviour changes "
            f"{t.when_to_change[0].lower() + t.when_to_change[1:]}. Effect of more uncertainty on the value of "
            f"flexibility: {t.more_uncertainty[0].lower() + t.more_uncertainty[1:]}."), "general"


def title(theory: str) -> str:
    return THEORY_BY_KEY[theory].title


def short(theory: str) -> str:
    return THEORY_BY_KEY[theory].title.split(":")[0]


def tournament_key(theory: str) -> Optional[str]:
    return THEORY_BY_KEY[theory].tournament_key
