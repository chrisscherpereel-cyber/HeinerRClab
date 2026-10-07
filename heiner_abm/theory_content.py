"""The nine theories of decision making under uncertainty that the laboratory tests, each described the same way.

Every theory has: origins and core idea, a formal core, what it says about behavioral flexibility under uncertainty,
how the laboratory implements it, strengths and limits, the hypotheses it informs, and key references. The pages in
app_pages/theory_*.py render these entries with ui.theory_page.render, so all theories get the same treatment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TheoryContent:
    key: str                       # also the agent-tournament theory key (heiner_abm.arena.KEYS)
    title: str
    icon: str
    tagline: str                   # one-sentence claim
    origins: str                   # markdown
    formal: str                    # markdown with LaTeX
    flexibility: str               # what the theory says about flexibility under uncertainty
    uncertainty_view: str          # one line for the overview table
    when_to_change: str            # one line: what triggers a change of behavior
    more_uncertainty: str          # one line: effect of more uncertainty on the value of flexibility
    strengths: Tuple[str, ...]
    limits: Tuple[str, ...]
    refs: Tuple[str, ...]          # keys in heiner_abm.literature.REFERENCES
    hypotheses: Tuple[str, ...]    # hypothesis IDs in the literature registry that this theory informs
    tournament_key: Optional[str]  # key in heiner_abm.theories (directional tournament), if the theory is in it
    page: str                      # app page file
    lab_notes: str = ""            # how the laboratory maps the theory's concepts onto the market (markdown)


THEORIES: List[TheoryContent] = [
    TheoryContent(
        "heiner", "Heiner: the reliability condition", "📘",
        "Imperfect agents do better by restricting behavior to simple rules whenever their deviations are not "
        "reliable enough, even when flexibility is free.",
        origins=(
            "Ronald Heiner (1983) argued that predictable, rule-governed behavior originates in uncertainty: when the "
            "difficulty of a problem exceeds the agent's competence (a *CD-gap*), attempts to exploit flexibility "
            "produce mistakes. He later formalized the dynamic version, partial adjustment toward an imperfectly "
            "perceived target (Heiner 1989), and linked the condition to signal-detection theory (Heiner 1986)."),
        formal=r"""
A default rule **B** has preferred exceptions with probability $\pi$. The agent deviates when an exception exists
with probability $r$ (hit rate) and when none exists with probability $w$ (false-alarm rate), gaining $G$ or losing $D$
on average. Flexibility pays only if

$$\underbrace{\frac{r}{w}}_{\text{reliability ratio}} \;>\; \underbrace{\frac{D}{G}\cdot\frac{1-\pi}{\pi}}_{\text{tolerance limit}}$$

As the CD-gap $U$ widens, $r(U)$ falls and $w(U)$ rises, so the condition eventually fails. In the dynamic version
(Heiner 1989) an agent moves a fraction $\beta$ toward a perceived target with error-to-signal ratio $K$; the maximal
reliable speed is $\beta_0 = 1/[(1+K)(1-f')]$, where $f'$ is the slope of the target map.""",
        flexibility="Flexibility is valuable only when deviations are reliable relative to their stakes. More "
                    "uncertainty makes restriction to rules optimal; behavior becomes more predictable.",
        uncertainty_view="A gap between the difficulty of the problem and the agent's competence",
        when_to_change="Only when deviations are reliable enough relative to their stakes",
        more_uncertainty="Lowers it: restrict behavior to rules",
        strengths=("Explains why free flexibility can hurt, which optimization cannot.",
                   "Combines accuracy (r, w) with stakes (π, G, D) in one testable condition.",
                   "Applies whatever the source of uncertainty."),
        limits=("Says when restriction pays, not how an agent can know its own r, w, π, G and D.",
                "Values a deviation as a one-shot bet; decisions in markets persist and provoke reactions."),
        refs=("heiner1983", "heiner1985", "heiner1986", "heiner1988", "heiner1988b", "heiner1989", "bookstaber1985"),
        hypotheses=("H1", "H2", "H3", "H4", "H7", "H8", "H9", "H10", "H11", "H12", "RC", "DRC", "BOUND", "SDT",
                    "KNIGHT", "PUNCT", "CDGAP", "EVO", "ARENA", "MECH1", "MECH2"),
        tournament_key="heiner", page="app_pages/theory_heiner.py",
        lab_notes="""
| Concept | In the cobweb market |
|---|---|
| **Difficulty** | Raw-material cost volatility Δ (firms commit output before the new cost is known), unannounced demand-regime shifts, and the number of rivals. |
| **Competence** | Cost foresight κ and perception noise σ; the measured CD-gap is the RMSE of each firm's cost perception. |
| **Rule B** | Keep producing last period's quantity. |
| **Flexible alternative** | The Cournot best reply (partial adjustment φ) or the margin-feedback quantity rule. |
| **Preferred exception** | A period in which adopting the recommendation earns more than following B, judged by forking the market over a horizon H. |
| **π, r, w, G, D** | Counted for every firm and every decision. |
| **Error-to-signal ratio K** | RMS error of the perceived target against the ex-post best reply, relative to the needed adjustment. |
"""),
    TheoryContent(
        "optimiser", "Neoclassical optimization", "📈",
        "A rational agent chooses the best action given all available information; more options can never hurt, and "
        "noise is handled by optimal forecasting.",
        origins=(
            "Optimization under uncertainty runs from Samuelson's *Foundations* (1947), whose Le Chatelier principle "
            "says that relaxing a constraint cannot lower the optimum, to Muth's (1961) rational expectations and to "
            "optimal filtering (Muth 1960; Kalman 1960). Brainard (1967) showed that uncertainty about the effect of "
            "an action makes the optimal response more cautious."),
        formal=r"""
**Le Chatelier.** If the constrained choice set $B$ is a subset of the flexible set $A$,

$$\max_{a\in A} \mathbb E[u(a,s)] \;\ge\; \max_{a\in B} \mathbb E[u(a,s)].$$

**Optimal filtering.** A target that follows a random walk (step variance $Q$) observed with noise (variance $R$) is
best tracked by exponential smoothing with the steady-state Kalman gain

$$K = \frac{P^-}{P^- + R},\qquad P^- = \tfrac12\big(Q + \sqrt{Q^2 + 4QR}\big),$$

which falls as the noise-to-signal ratio $R/Q$ rises. **Brainard attenuation:** to hit a target $y^*$ when the effect
$b$ of action $x$ is uncertain (mean $\mu$, variance $\sigma^2$), the optimal action is $x^* = \mu y^*/(\mu^2 + \sigma^2)$.""",
        flexibility="Free flexibility never lowers the optimum. Under noise the optimizer responds less to each "
                    "signal (attenuation), but it never restricts itself to a fixed rule.",
        uncertainty_view="Risk with known (or learnable) probabilities, handled by expectations",
        when_to_change="Whenever the expected payoff of changing is positive",
        more_uncertainty="Never negative; responses are attenuated, not restricted",
        strengths=("Precise, general benchmark with closed-form predictions.",
                   "Optimal filtering explains partial adjustment without appeal to decision errors."),
        limits=("Assumes the model of the world is correct; misspecification is outside the theory.",
                "Treats the cost of computing and knowing the optimum as zero."),
        refs=("samuelson1947", "muth1960", "muth1961", "kalman1960", "brainard1967"),
        hypotheses=("H1", "H4", "H10", "RC", "DRC", "BOUND", "MECH1"),
        tournament_key="neo", page="app_pages/theory_optimiser.py",
        lab_notes="The optimizer's designs share the model-based target with the reliability-condition and "
                  "inaction-band designs, so the comparison isolates *when* to act from *where* to aim."),
    TheoryContent(
        "options", "Real options: the value of flexibility", "🔀",
        "Flexibility is an option whose value rises with uncertainty; with adjustment costs, the option to wait makes "
        "agents act only on large gaps.",
        origins=(
            "Stigler (1939) showed that plants built for flexibility trade static efficiency for adaptability. "
            "Marschak & Nelson (1962), Jones & Ostroy (1984) and Kreps (1979) formalized the value of flexibility "
            "under uncertainty. Dixit (1989) and Dixit & Pindyck (1994) brought option pricing to real decisions, "
            "showing hysteresis: with costs of adjustment there is a zone of inaction that widens with volatility, "
            "as in Scarf's (1960) (S, s) policies."),
        formal=r"""
The value $F$ of an option to act later rises with the volatility $\sigma$ of the underlying:
$\partial F/\partial\sigma > 0$. With a fixed cost $k$ of adjusting and a loss that grows with the gap between the
current position $x$ and the target $x^*$, the optimal policy is an **inaction band**:

$$\text{adjust to } x^* \text{ only if } |x - x^*| > b^*(\sigma, k),$$

with $b^*$ increasing in both $\sigma$ and $k$ (hysteresis).""",
        flexibility="Uncertainty raises the value of keeping options open; with adjustment costs it also widens "
                    "the zone of inaction, so agents act less often but more decisively.",
        uncertainty_view="Volatility of the payoff-relevant state (risk)",
        when_to_change="When the gap to the target exceeds an inaction band",
        more_uncertainty="Raises the value of the option; widens the inaction band",
        strengths=("Explains inaction and lumpy adjustment from optimization.",
                   "Quantifies the value of flexibility and its dependence on volatility."),
        limits=("The band comes from adjustment costs, not from decision errors.",
                "Like optimization, assumes the stochastic process is known."),
        refs=("stigler1939", "marschak1962", "kreps1979", "jones1984", "dixit1989", "dixit1994", "scarf1960",
              "caballero1999"),
        hypotheses=("H3", "H7", "H8", "H9", "H12", "KNIGHT", "MECH2"),
        tournament_key="options", page="app_pages/theory_options.py",
        lab_notes="The laboratory implements an **inaction-band heuristic inspired by real options**: the band's "
                  "width scales with the measured volatility of the agent's own target, echoing the real-options "
                  "logic of acting only when the gap is large relative to uncertainty. It is not a dynamic "
                  "real-options model: there is no adjustment cost, no irreversibility and no option value computed "
                  "from a known process; the band width is a tuned parameter. The signature test compares an "
                  "always-adjusting firm with its rigid twin, without adjustment costs, so it tests whether "
                  "flexibility gains value with volatility in this market, not real-options theory as a whole."),
    TheoryContent(
        "cobweb", "Cobweb theory and adaptive expectations", "🕸️",
        "Fast adjustment to the latest signal destabilizes markets; slow, adaptive adjustment stabilizes them.",
        origins=(
            "Ezekiel's (1938) cobweb theorem explained cycles in agricultural markets by producers who commit output "
            "on last period's price. Nerlove (1958) replaced naive with adaptive expectations, and Theocharis (1960) "
            "showed that Cournot best-reply adjustment becomes unstable as the number of firms grows. Laboratory "
            "cobweb markets (Carlson 1967; Hommes et al. 2007) and heterogeneous-expectation models (Brock & Hommes "
            "1997) continue the line."),
        formal=r"""
Demand $Q^d_t = a - dP_t$, supply $Q^s_t = c + bP^e_t$, adaptive expectations
$P^e_t = P^e_{t-1} + \lambda(P_{t-1} - P^e_{t-1})$. The expected price evolves as
$P^e_{t+1} = [1 - \lambda(1 + b/d)]P^e_t + \text{const}$, so the market converges if and only if

$$\frac{b}{d} \;<\; \frac{2-\lambda}{\lambda}\qquad(\text{Nerlove 1958; } \lambda = 1 \text{ gives Ezekiel's } b/d < 1).$$

With $n$ Cournot firms each moving a share $\varphi$ toward its best reply, the market is stable only if
$\varphi < 4/(n+1)$ (Theocharis 1960 for $\varphi = 1$).""",
        flexibility="Flexibility (adjustment speed) hurts only by destabilizing the market; additive noise does not "
                    "change stability. Slow, adaptive adjustment is safe.",
        uncertainty_view="Expectations about the next price, formed from past prices",
        when_to_change="Every period, by a fraction of the latest forecast error",
        more_uncertainty="No direct effect; stability depends on adjustment speed and market structure",
        strengths=("Explains cycles and instability from a simple, observable rule.",
                   "Model-free: needs only observed prices, so it is robust to a misspecified demand model."),
        limits=("Says little about which agent benefits from flexibility, only about market stability.",
                "Adaptive expectations ignore known structure that agents could use."),
        refs=("ezekiel1938", "nerlove1958", "theocharis1960", "carlson1967", "carlson1968", "hommes1994", "brock1997",
              "hommes2007"),
        hypotheses=("H6", "H11", "BOUND", "PUNCT", "MECH2"),
        tournament_key="cobweb", page="app_pages/theory_cobweb.py",
        lab_notes="The laboratory's market is itself a cobweb oligopoly: firms commit output before the cost and price "
                  "are known. Its stability index (the spectral radius of the linearized production dynamics) is "
                  "computed for every environment and used as cobweb theory's forecast on the *Competing theories* "
                  "page."),
    TheoryContent(
        "heuristic", "Simple heuristics and the bias–variance trade-off", "🎯",
        "Simple rules that ignore information can beat flexible models when data are noisy or scarce; which rule works "
        "depends on the structure of the environment.",
        origins=(
            "Simon (1956) argued that rational behavior is shaped jointly by the environment and the agent's "
            "capacities. Dawes (1979) showed that equal-weight linear models predict as well as fitted ones. The "
            "ecological-rationality program (Gigerenzer, Todd & ABC Research Group 1999; Gigerenzer & Brighton "
            "2009) explains such *less-is-more* effects with the bias–variance trade-off (Geman et al. 1992). Full-cost "
            "and target-margin pricing rules (Hall & Hitch 1939) are business examples."),
        formal=r"""
For a prediction $\hat y$ of $y = f(x) + \varepsilon$, the expected squared error decomposes as

$$\mathbb E\big[(\hat y - y)^2\big] = \underbrace{\big(\mathbb E[\hat y] - f\big)^2}_{\text{bias}^2} +
\underbrace{\mathrm{Var}[\hat y]}_{\text{variance}} + \sigma^2_\varepsilon .$$

Flexible models have low bias but high variance; simple rules accept bias for low variance. When samples are small or
noise is large, the variance term dominates and the simple rule wins.""",
        flexibility="Flexible models overfit noise. Simple rules win when estimation error is large; only accuracy "
                    "matters, not the stakes of mistakes.",
        uncertainty_view="Estimation error from limited, noisy experience",
        when_to_change="By a fixed rule that ignores much of the available information",
        more_uncertainty="Lowers it: noise favors simple rules",
        strengths=("Explains when ignoring information is smart, with a precise statistical mechanism.",
                   "Points to the match between rule and environment rather than to a universal best rule."),
        limits=("Silent on the stakes of mistakes, which the reliability condition includes.",
                "Which heuristic suits which environment must be found case by case."),
        refs=("simon1956", "gigerenzer1999", "gigerenzer2009", "geman1992", "dawes1979", "makridakis2000", "hall1939"),
        hypotheses=("H1", "H3", "H8", "H9"),
        tournament_key="biasvar", page="app_pages/theory_heuristics.py"),
    TheoryContent(
        "satisficing", "Satisficing and aspiration-level search", "🎚️",
        "Decision makers keep what works and search for change only when performance falls short of an adaptive "
        "aspiration level.",
        origins=(
            "Simon (1955) replaced maximization with satisficing: accept the first option that meets an aspiration. "
            "Cyert & March (1963) built the behavioral theory of the firm on problemistic search triggered by "
            "performance below aspirations, which adapt to experience. Greve (1998) found that shortfalls raise the "
            "likelihood of risky organizational change; March & Shapira (1987) showed that attention to survival "
            "points shapes risk taking. Lindblom (1959) described policy as incremental *muddling through*."),
        formal=r"""
With performance $\pi_t$ and aspiration $A_t$,

$$\text{search (change behavior) if } \pi_t < A_t,\quad\text{keep behavior otherwise};\qquad
A_{t+1} = (1-\alpha)A_t + \alpha\,\pi_t .$$

Search intensity rises with the shortfall $A_t - \pi_t$; the aspiration adapts at speed $\alpha$.""",
        flexibility="Change is triggered by failure, not by opportunity. More volatile performance pushes results below "
                    "aspiration more often, so it triggers more search and change.",
        uncertainty_view="Unpredictable performance relative to an aspiration",
        when_to_change="When performance falls below an adaptive aspiration",
        more_uncertainty="Increases search and change",
        strengths=("Describes how organizations actually decide, with strong empirical support.",
                   "Needs no model of the environment."),
        limits=("Descriptive: it predicts what firms do, not what pays.",
                "Change triggered by noise in performance can be change for the worse."),
        refs=("simon1955", "cyert1963", "march1987", "greve1998", "lindblom1959"),
        hypotheses=("H2", "H9", "H10"),
        tournament_key="satisficing", page="app_pages/theory_satisficing.py"),
    TheoryContent(
        "rl", "Reinforcement learning", "🧠",
        "Agents learn which actions pay from the payoffs that followed them, balancing exploration against "
        "exploitation.",
        origins=(
            "Reinforcement learning models in economics (Roth & Erev 1995; Erev & Roth 1998) predict behavior in "
            "repeated games from propensities that grow with past payoffs. Computer science formalized value-based "
            "learning and the exploration–exploitation trade-off (Sutton & Barto 2018). In organizations, March (1991) "
            "and Levinthal & March (1993) showed how learning from experience can become myopic."),
        formal=r"""
Value learning with softmax choice:

$$V_{t+1}(a) = V_t(a) + \eta\,\big(r_t - V_t(a)\big),\qquad P_t(a) = \frac{e^{V_t(a)/\tau}}{\sum_{a'} e^{V_t(a')/\tau}} .$$

Erev–Roth propensities: $q_{t+1}(a) = (1-\phi)\,q_t(a) + R_t\,\mathbf 1[a = a_t]$, with $P_t(a) \propto q_t(a)$.
The learning rate $\eta$ (or forgetting $\phi$) trades responsiveness against noise; the temperature $\tau$ trades
exploration against exploitation.""",
        flexibility="Flexibility is learned: actions that paid are repeated. In changing environments, learning must "
                    "forget, which makes estimates noisy.",
        uncertainty_view="Unknown payoffs, learned from experience",
        when_to_change="When learned values favor another action (plus exploration)",
        more_uncertainty="Raises the need to explore and to forget; learning becomes noisier",
        strengths=("Needs no model of the environment and adapts to whatever payoffs it meets.",
                   "Fits laboratory behavior in repeated games well."),
        limits=("Learns slowly; in non-stationary environments its estimates stay noisy.",
                "Rewards depend on the agent's own past actions, which can lock in poor habits."),
        refs=("erev1998", "roth1995", "sutton2018", "march1991", "levinthal1993"),
        hypotheses=("RC", "ARENA"),
        tournament_key=None, page="app_pages/theory_rl.py"),
    TheoryContent(
        "imitation", "Imitation and evolutionary selection", "🧬",
        "No one needs to know what is optimal: successful behavior spreads by imitation and selection.",
        origins=(
            "Alchian (1950) argued that selection among firms can produce adaptive outcomes without optimization. "
            "Nelson & Winter (1982) modeled firms as bundles of routines that change slowly. Hannan & Freeman (1984) "
            "showed that selection favors structurally inert, reliable organizations. Vega-Redondo (1997) proved "
            "that imitating the most profitable firm drives Cournot markets to the competitive (Walrasian) output, "
            "which experiments confirm (Huck et al. 1999); Boyd & Richerson (1985) analyzed conformist transmission."),
        formal=r"""
**Imitate the best.** Each period a firm copies the output of the most profitable firm, plus small experimentation:
$q_{i,t+1} = q_{j^*,t} + \varepsilon_{i,t}$, $j^* = \arg\max_j \pi_{j,t}$. With linear demand $P = a - bQ$ and constant
cost $c$, the long-run (stochastically stable) outcome is the Walrasian output

$$Q^W = \frac{a-c}{b} \quad\text{rather than the Cournot–Nash output}\quad Q^N = \frac{n}{n+1}\cdot\frac{a-c}{b}.$$

**Replicator dynamics:** the share $x_k$ of behavior $k$ grows with its payoff advantage, $\dot x_k = x_k(\pi_k - \bar\pi)$.""",
        flexibility="Behavior changes by copying success, not by calculation. Imitation can make markets more "
                    "aggressive. (Selection for reliable, inert organizations is the separate theory of "
                    "organizational ecology.)",
        uncertainty_view="Unknown best behavior, inferred from others' success",
        when_to_change="When another firm is doing better (imitation), or through selection",
        more_uncertainty="No direct prediction; noise blurs who is successful",
        strengths=("Explains population-level outcomes without assuming individual optimization.",
                   "Predicts convergence to competitive outcomes under imitation, with experimental support."),
        limits=("Copies relative success, which can differ from absolute performance.",
                "Says little about when an individual should be flexible."),
        refs=("alchian1950", "nelson1982", "vegaredondo1997", "huck1999", "boyd1985"),
        hypotheses=("H5", "H6", "EVO", "ARENA"),
        tournament_key=None, page="app_pages/theory_imitation.py"),
    TheoryContent(
        "ecology", "Organizational ecology: structural inertia", "🏛️",
        "Selection favors reliable, accountable organizations, and reliability requires structural inertia; "
        "change itself is hazardous, so inert organizations survive in any environment.",
        origins=(
            "Hannan & Freeman (1977) moved the question of adaptation from the single organization to the population: "
            "environments select among organizational forms. Hannan & Freeman (1984) argued that selection favors "
            "organizations that perform reliably and can account for their actions, that reliability and "
            "accountability require routines that are hard to change (structural inertia), and that attempts at "
            "core change raise the risk of failure. Amburgey, Kelly & Barnett (1993) found that change resets the "
            "liability of newness: failure rates rise after reorganization."),
        formal=r"""
Organizations of form $k$ fail at rate $\mu_k$ and are founded at rate $\lambda_k$; the population of each form evolves as
$\dot N_k = (\lambda_k - \mu_k)N_k$. Inertia lowers the variance of performance, which lowers the failure rate when
capital is a buffer: with per-period performance $x_t \sim (m, s^2)$ and capital $C$, the probability of ruin falls with
$C\,m/s^2$. Reorganization at time $t_0$ raises the hazard temporarily:

$$\mu(t) = \mu_0 + \Delta\mu\, e^{-(t - t_0)/\tau}\quad (t \ge t_0),$$

so a change pays only if its expected gain outweighs the extra risk of failure it creates.""",
        flexibility="Flexibility is hazardous: each reorganization exposes the organization to failure, and reliable "
                    "performance, which selection rewards, requires inert routines. Inertia is selected in volatile "
                    "and calm environments alike.",
        uncertainty_view="Unknown fitness of organizational forms, revealed by selection",
        when_to_change="Rarely: on a schedule set by routines, or when failure threatens",
        more_uncertainty="No change in the ranking: inertia is favored everywhere",
        strengths=("Explains why established organizations change slowly even when change looks profitable.",
                   "Links reliability of performance to survival, a selection argument that needs no optimization."),
        limits=("Predicts at the level of populations; says little about which single decisions should change.",
                "Hard to distinguish inertia that is selected from inertia that is merely costly to overcome."),
        refs=("hannan1977", "hannan1984", "amburgey1993", "nelson1982"),
        hypotheses=("H1", "EVO", "ARENA"),
        tournament_key="ecology", page="app_pages/theory_ecology.py",
        lab_notes=("Ecology's agents keep their routine output and reorganize only rarely: on a fixed schedule "
                   "(*scheduled reorganization*) or when smoothed profit falls below a survival threshold "
                   "(*reorganize under threat of failure*). A reorganization moves output toward the price-based "
                   "target, the same target real-options, satisficing and Heiner agents can use, so ecology is "
                   "compared on the timing of change, not on a better forecast.")),
]
THEORY_BY_KEY: Dict[str, TheoryContent] = {t.key: t for t in THEORIES}
