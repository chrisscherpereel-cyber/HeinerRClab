"""Research basis of the simulation: references, the literature behind every hypothesis and its alternative, and
the contribution each test makes.

Every hypothesis tested in the app has
    * supporting research for the reliability-condition (RC) prediction,
    * supporting research for the alternative (rival) prediction, and
    * a statement of what the simulation adds to that literature.

References are stored once (REFERENCES) and cited by key, so the bibliography, the in-app citations and the
BibTeX export stay consistent. `tests/test_literature.py` checks that every cited key exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Ref:
    key: str
    authors: str                 # APA author list, e.g. "Heiner, R. A."
    year: int
    title: str
    kind: str                    # "article", "book", "chapter" or "preprint" (journal holds the repository id)
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    booktitle: str = ""
    editors: str = ""

    @property
    def cite(self) -> str:
        """Short in-text citation, e.g. 'Heiner (1983)' or 'Dixit & Pindyck (1994)'."""
        names = [a.split(",")[0].strip() for a in self.authors.split("; ")]
        if len(names) == 1:
            who = names[0]
        elif len(names) == 2:
            who = f"{names[0]} & {names[1]}"
        else:
            who = f"{names[0]} et al."
        return f"{who} ({self.year})"

    @property
    def apa(self) -> str:
        """APA-style reference (Markdown italics)."""
        parts = self.authors.split("; ")
        authors = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", & " + parts[-1]
        if self.kind == "article":
            vol = f", *{self.volume}*" if self.volume else ""
            iss = f"({self.issue})" if self.issue else ""
            pg = f", {self.pages}" if self.pages else ""
            return f"{authors} ({self.year}). {self.title}. *{self.journal}*{vol}{iss}{pg}."
        if self.kind == "preprint":
            return f"{authors} ({self.year}). {self.title}. *{self.journal}*."
        if self.kind == "chapter":
            multi = "&" in self.editors or "," in self.editors
            ed = f"In {self.editors} ({'Eds.' if multi else 'Ed.'}), " if self.editors else "In "
            pg = f" (pp. {self.pages})" if self.pages else ""
            return f"{authors} ({self.year}). {self.title}. {ed}*{self.booktitle}*{pg}. {self.publisher}."
        return f"{authors} ({self.year}). *{self.title}*. {self.publisher}."

    @property
    def bibtex(self) -> str:
        authors = " and ".join(self.authors.split("; "))
        f = {"author": authors, "year": str(self.year), "title": self.title}
        if self.kind == "article":
            f.update(journal=self.journal, volume=self.volume, number=self.issue, pages=self.pages.replace("–", "--"))
            typ = "article"
        elif self.kind == "preprint":
            f.update(howpublished=self.journal)
            typ = "misc"
        elif self.kind == "chapter":
            f.update(booktitle=self.booktitle, editor=self.editors.replace(" & ", " and ").replace(", ", " and "), publisher=self.publisher,
                     pages=self.pages.replace("–", "--"))
            typ = "incollection"
        else:
            f.update(publisher=self.publisher)
            typ = "book"
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)
        return f"@{typ}{{{self.key},\n{body}\n}}"


def _a(key, authors, year, title, journal, volume, issue, pages):
    return Ref(key, authors, year, title, "article", journal=journal, volume=volume, issue=issue, pages=pages)


def _b(key, authors, year, title, publisher):
    return Ref(key, authors, year, title, "book", publisher=publisher)


def _c(key, authors, year, title, booktitle, editors, pages, publisher):
    return Ref(key, authors, year, title, "chapter", booktitle=booktitle, editors=editors, pages=pages,
               publisher=publisher)


_REFS = [
    # --- Heiner and rule-governed behaviour
    _a("heiner1983", "Heiner, R. A.", 1983, "The origin of predictable behavior", "American Economic Review",
       "73", "4", "560–595"),
    _a("heiner1985", "Heiner, R. A.", 1985, "Origin of predictable behavior: Further modeling and applications",
       "American Economic Review", "75", "2", "391–396"),
    _c("heiner1986", "Heiner, R. A.", 1986, "Uncertainty, signal-detection experiments, and modeling behavior",
       "Economics as a process: Essays in the new institutional economics", "R. N. Langlois", "59–115",
       "Cambridge University Press"),
    _a("heiner1988", "Heiner, R. A.", 1988, "The necessity of imperfect decisions",
       "Journal of Economic Behavior & Organization", "10", "1", "29–55"),
    _a("heiner1988b", "Heiner, R. A.", 1988, "The necessity of delaying economic adjustment",
       "Journal of Economic Behavior & Organization", "10", "3", "255–286"),
    _a("heiner1989", "Heiner, R. A.", 1989, "The origin of predictable dynamic behavior",
       "Journal of Economic Behavior & Organization", "12", "2", "233–257"),
    _a("bookstaber1985", "Bookstaber, R.; Langsam, J.", 1985, "On the optimality of coarse behavior rules",
       "Journal of Theoretical Biology", "116", "2", "161–193"),
    _a("conlisk1996", "Conlisk, J.", 1996, "Why bounded rationality?", "Journal of Economic Literature", "34", "2",
       "669–700"),
    _b("friedman1960", "Friedman, M.", 1960, "A program for monetary stability", "Fordham University Press"),
    _a("kydland1977", "Kydland, F. E.; Prescott, E. C.", 1977,
       "Rules rather than discretion: The inconsistency of optimal plans", "Journal of Political Economy", "85", "3",
       "473–491"),
    # --- bounded rationality, behavioural theory of the firm, routines
    _a("simon1955", "Simon, H. A.", 1955, "A behavioral model of rational choice", "Quarterly Journal of Economics",
       "69", "1", "99–118"),
    _a("simon1956", "Simon, H. A.", 1956, "Rational choice and the structure of the environment",
       "Psychological Review", "63", "2", "129–138"),
    _b("cyert1963", "Cyert, R. M.; March, J. G.", 1963, "A behavioral theory of the firm", "Prentice-Hall"),
    _a("greve1998", "Greve, H. R.", 1998, "Performance, aspirations, and risky organizational change",
       "Administrative Science Quarterly", "43", "1", "58–86"),
    _a("lindblom1959", "Lindblom, C. E.", 1959, "The science of \"muddling through\"", "Public Administration Review",
       "19", "2", "79–88"),
    _b("nelson1982", "Nelson, R. R.; Winter, S. G.", 1982, "An evolutionary theory of economic change",
       "Belknap Press of Harvard University Press"),
    _a("alchian1950", "Alchian, A. A.", 1950, "Uncertainty, evolution, and economic theory",
       "Journal of Political Economy", "58", "3", "211–221"),
    _a("march1991", "March, J. G.", 1991, "Exploration and exploitation in organizational learning",
       "Organization Science", "2", "1", "71–87"),
    _a("levinthal1993", "Levinthal, D. A.; March, J. G.", 1993, "The myopia of learning",
       "Strategic Management Journal", "14", "S2", "95–112"),
    _a("hannan1984", "Hannan, M. T.; Freeman, J.", 1984, "Structural inertia and organizational change",
       "American Sociological Review", "49", "2", "149–164"),
    _a("hannan1977", "Hannan, M. T.; Freeman, J.", 1977, "The population ecology of organizations",
       "American Journal of Sociology", "82", "5", "929–964"),
    _a("amburgey1993", "Amburgey, T. L.; Kelly, D.; Barnett, W. P.", 1993,
       "Resetting the clock: The dynamics of organizational change and failure", "Administrative Science Quarterly",
       "38", "1", "51–73"),
    _b("thompson1967", "Thompson, J. D.", 1967, "Organizations in action", "McGraw-Hill"),
    _a("staw1981", "Staw, B. M.; Sandelands, L. E.; Dutton, J. E.", 1981,
       "Threat-rigidity effects in organizational behavior: A multilevel analysis",
       "Administrative Science Quarterly", "26", "4", "501–524"),
    _a("march1987", "March, J. G.; Shapira, Z.", 1987, "Managerial perspectives on risk and risk taking",
       "Management Science", "33", "11", "1404–1418"),
    _a("kahneman1979", "Kahneman, D.; Tversky, A.", 1979, "Prospect theory: An analysis of decision under risk",
       "Econometrica", "47", "2", "263–291"),
    _a("leonardbarton1992", "Leonard-Barton, D.", 1992,
       "Core capabilities and core rigidities: A paradox in managing new product development",
       "Strategic Management Journal", "13", "S1", "111–125"),
    _a("eisenhardt2000", "Eisenhardt, K. M.; Martin, J. A.", 2000, "Dynamic capabilities: What are they?",
       "Strategic Management Journal", "21", "10–11", "1105–1121"),
    _a("teece1997", "Teece, D. J.; Pisano, G.; Shuen, A.", 1997, "Dynamic capabilities and strategic management",
       "Strategic Management Journal", "18", "7", "509–533"),
    _a("volberda1996", "Volberda, H. W.", 1996,
       "Toward the flexible form: How to remain vital in hypercompetitive environments", "Organization Science", "7",
       "4", "359–374"),
    # --- heuristics, bias-variance, forecasting
    _a("gigerenzer2009", "Gigerenzer, G.; Brighton, H.", 2009,
       "Homo heuristicus: Why biased minds make better inferences", "Topics in Cognitive Science", "1", "1",
       "107–143"),
    _b("gigerenzer1999", "Gigerenzer, G.; Todd, P. M.; ABC Research Group", 1999, "Simple heuristics that make us smart",
       "Oxford University Press"),
    _a("geman1992", "Geman, S.; Bienenstock, E.; Doursat, R.", 1992, "Neural networks and the bias/variance dilemma",
       "Neural Computation", "4", "1", "1–58"),
    _a("dawes1979", "Dawes, R. M.", 1979, "The robust beauty of improper linear models in decision making",
       "American Psychologist", "34", "7", "571–582"),
    _a("makridakis2000", "Makridakis, S.; Hibon, M.", 2000, "The M3-Competition: Results, conclusions and implications",
       "International Journal of Forecasting", "16", "4", "451–476"),
    # --- optimisation, flexibility and real options
    _b("samuelson1947", "Samuelson, P. A.", 1947, "Foundations of economic analysis", "Harvard University Press"),
    _a("muth1961", "Muth, J. F.", 1961, "Rational expectations and the theory of price movements", "Econometrica", "29",
       "3", "315–335"),
    _a("muth1960", "Muth, J. F.", 1960, "Optimal properties of exponentially weighted forecasts",
       "Journal of the American Statistical Association", "55", "290", "299–306"),
    _a("kalman1960", "Kalman, R. E.", 1960, "A new approach to linear filtering and prediction problems",
       "Journal of Basic Engineering", "82", "1", "35–45"),
    _a("brainard1967", "Brainard, W. C.", 1967, "Uncertainty and the effectiveness of policy", "American Economic Review",
       "57", "2", "411–425"),
    _a("stigler1939", "Stigler, G.", 1939, "Production and distribution in the short run", "Journal of Political Economy",
       "47", "3", "305–327"),
    _a("marschak1962", "Marschak, T.; Nelson, R.", 1962, "Flexibility, uncertainty, and economic theory",
       "Metroeconomica", "14", "1–3", "42–58"),
    _a("jones1984", "Jones, R. A.; Ostroy, J. M.", 1984, "Flexibility and uncertainty", "Review of Economic Studies",
       "51", "1", "13–32"),
    _a("kreps1979", "Kreps, D. M.", 1979, "A representation theorem for \"preference for flexibility\"", "Econometrica",
       "47", "3", "565–577"),
    _a("dixit1989", "Dixit, A.", 1989, "Entry and exit decisions under uncertainty", "Journal of Political Economy", "97",
       "3", "620–638"),
    _b("dixit1994", "Dixit, A. K.; Pindyck, R. S.", 1994, "Investment under uncertainty", "Princeton University Press"),
    _c("scarf1960", "Scarf, H.", 1960, "The optimality of (S, s) policies in the dynamic inventory problem",
       "Mathematical methods in the social sciences, 1959", "K. J. Arrow, S. Karlin & P. Suppes", "196–202",
       "Stanford University Press"),
    _a("sheshinski1977", "Sheshinski, E.; Weiss, Y.", 1977, "Inflation and costs of price adjustment",
       "Review of Economic Studies", "44", "2", "287–303"),
    _a("caballero1999", "Caballero, R. J.; Engel, E. M. R. A.", 1999,
       "Explaining investment dynamics in U.S. manufacturing: A generalized (S, s) approach", "Econometrica", "67", "4",
       "783–826"),
    # --- uncertainty, ambiguity, robustness
    _b("knight1921", "Knight, F. H.", 1921, "Risk, uncertainty and profit", "Houghton Mifflin"),
    _a("arrow1951", "Arrow, K. J.", 1951, "Alternative approaches to the theory of choice in risk-taking situations",
       "Econometrica", "19", "4", "404–437"),
    _a("lucas1977", "Lucas, R. E., Jr.", 1977, "Understanding business cycles",
       "Carnegie-Rochester Conference Series on Public Policy", "5", "", "7–29"),
    _a("gilboa1989", "Gilboa, I.; Schmeidler, D.", 1989, "Maxmin expected utility with non-unique prior",
       "Journal of Mathematical Economics", "18", "2", "141–153"),
    _a("hansen2001", "Hansen, L. P.; Sargent, T. J.", 2001, "Robust control and model uncertainty",
       "American Economic Review", "91", "2", "60–66"),
    _b("hansen2008", "Hansen, L. P.; Sargent, T. J.", 2008, "Robustness", "Princeton University Press"),
    # --- cobweb dynamics, oligopoly learning, collusion
    _a("ezekiel1938", "Ezekiel, M.", 1938, "The cobweb theorem", "Quarterly Journal of Economics", "52", "2", "255–280"),
    _a("nerlove1958", "Nerlove, M.", 1958, "Adaptive expectations and cobweb phenomena",
       "Quarterly Journal of Economics", "72", "2", "227–240"),
    _a("theocharis1960", "Theocharis, R. D.", 1960, "On the stability of the Cournot solution on the oligopoly problem",
       "Review of Economic Studies", "27", "2", "133–134"),
    _a("carlson1967", "Carlson, J. A.", 1967, "The stability of an experimental market with a supply-response lag",
       "Southern Economic Journal", "33", "3", "305–321"),
    _a("carlson1968", "Carlson, J. A.", 1968, "An invariably stable cobweb model", "Review of Economic Studies", "35",
       "3", "360–362"),
    _a("hommes1994", "Hommes, C. H.", 1994,
       "Dynamics of the cobweb model with adaptive expectations and nonlinear supply and demand",
       "Journal of Economic Behavior & Organization", "24", "3", "315–335"),
    _a("brock1997", "Brock, W. A.; Hommes, C. H.", 1997, "A rational route to randomness", "Econometrica", "65", "5",
       "1059–1095"),
    _a("sonnemans2004", "Sonnemans, J.; Hommes, C.; Tuinstra, J.; van de Velden, H.", 2004,
       "The instability of a heterogeneous cobweb economy: A strategy experiment on expectation formation",
       "Journal of Economic Behavior & Organization", "54", "4", "453–481"),
    _a("hommes2007", "Hommes, C.; Sonnemans, J.; Tuinstra, J.; van de Velden, H.", 2007, "Learning in cobweb experiments",
       "Macroeconomic Dynamics", "11", "S1", "8–33"),
    _a("vegaredondo1997", "Vega-Redondo, F.", 1997, "The evolution of Walrasian behavior", "Econometrica", "65", "2",
       "375–384"),
    _a("huck1999", "Huck, S.; Normann, H.-T.; Oechssler, J.", 1999, "Learning in Cournot oligopoly: An experiment",
       "Economic Journal", "109", "454", "C80–C95"),
    _a("hommes2011", "Hommes, C.", 2011, "The heterogeneous expectations hypothesis: Some evidence from the lab",
       "Journal of Economic Dynamics and Control", "35", "1", "1–24"),
    _a("anufriev2012", "Anufriev, M.; Hommes, C.", 2012, "Evolutionary selection of individual expectations and "
       "aggregate outcomes in asset pricing experiments", "American Economic Journal: Microeconomics", "4", "4",
       "35–64"),
    _a("grimm2005", "Grimm, V.; Revilla, E.; Berger, U.; Jeltsch, F.; Mooij, W. M.; Railsback, S. F.; Thulke, H.-H.; "
       "Weiner, J.; Wiegand, T.; DeAngelis, D. L.", 2005, "Pattern-oriented modeling of agent-based complex systems: "
       "Lessons from ecology", "Science", "310", "5750", "987–991"),
    _a("arrow1951b", "Arrow, K. J.; Harris, T.; Marschak, J.", 1951, "Optimal inventory policy", "Econometrica", "19",
       "3", "250–272"),
    _a("schweitzer2000", "Schweitzer, M. E.; Cachon, G. P.", 2000, "Decision bias in the newsvendor problem with a "
       "known demand distribution: Experimental evidence", "Management Science", "46", "3", "404–420"),
    _a("bolton2008", "Bolton, G. E.; Katok, E.", 2008, "Learning by doing in the newsvendor problem: A laboratory "
       "investigation of the role of experience and feedback", "Manufacturing & Service Operations Management",
       "10", "3", "519–538"),
    _a("behrens2007", "Behrens, T. E. J.; Woolrich, M. W.; Walton, M. E.; Rushworth, M. F. S.", 2007,
       "Learning the value of information in an uncertain world", "Nature Neuroscience", "10", "9", "1214–1221"),
    _a("fischbacher2007", "Fischbacher, U.", 2007, "z-Tree: Zurich toolbox for ready-made economic experiments",
       "Experimental Economics", "10", "2", "171–178"),
    _a("gomezmartinez2016", "Gomez-Martinez, F.; Onderstal, S.; Sonnemans, J.", 2016,
       "Firm-specific information and explicit collusion in experimental oligopolies", "European Economic Review",
       "82", "", "132–141"),
    _a("evans2025", "Evans, G. W.; Gibbs, C. G.; McGough, B.", 2025, "A unified model of learning to forecast",
       "American Economic Journal: Macroeconomics", "17", "2", "101–133"),
    _a("brokesova2022", "Brokesova, Z.; Deck, C.; Peliova, J.", 2022, "Pull-to-center is not just for newsvendors",
       "PLOS ONE", "17", "2", "e0264183"),
    _a("hommes2005", "Hommes, C.; Sonnemans, J.; Tuinstra, J.; van de Velden, H.", 2005,
       "Coordination of expectations in asset pricing experiments", "Review of Financial Studies", "18", "3",
       "955–980"),
    _a("hommes2008", "Hommes, C.; Sonnemans, J.; Tuinstra, J.; van de Velden, H.", 2008,
       "Expectations and bubbles in asset pricing experiments", "Journal of Economic Behavior & Organization", "67",
       "1", "116–133"),
    _a("bostian2008", "Bostian, A. A.; Holt, C. A.; Smith, A. M.", 2008,
       "Newsvendor \"pull-to-center\" effect: Adaptive learning in a laboratory experiment",
       "Manufacturing & Service Operations Management", "10", "4", "590–608"),
    _b("creed2021", "CREED, University of Amsterdam", 2021,
       "Pushed to perform: Time pressure in long run Learning-to-Forecast experiments [Data set]",
       "University of Amsterdam figshare, article 13948409"),
    _a("huck2004", "Huck, S.; Normann, H.-T.; Oechssler, J.", 2004,
       "Two are few and four are many: Number effects in experimental oligopolies",
       "Journal of Economic Behavior & Organization", "53", "4", "435–446"),
    _a("stigler1964", "Stigler, G. J.", 1964, "A theory of oligopoly", "Journal of Political Economy", "72", "1", "44–61"),
    _a("green1984", "Green, E. J.; Porter, R. H.", 1984, "Noncooperative collusion under imperfect price information",
       "Econometrica", "52", "1", "87–100"),
    _a("carlton1986", "Carlton, D. W.", 1986, "The rigidity of prices", "American Economic Review", "76", "4", "637–658"),
    _a("nickell1996", "Nickell, S. J.", 1996, "Competition and corporate performance", "Journal of Political Economy",
       "104", "4", "724–746"),
    _a("aghion2005", "Aghion, P.; Bloom, N.; Blundell, R.; Griffith, R.; Howitt, P.", 2005,
       "Competition and innovation: An inverted-U relationship", "Quarterly Journal of Economics", "120", "2",
       "701–728"),
    # --- learning
    _a("erev1998", "Erev, I.; Roth, A. E.", 1998,
       "Predicting how people play games: Reinforcement learning in experimental games with unique, mixed strategy "
       "equilibria", "American Economic Review", "88", "4", "848–881"),
    # --- decision benchmarks (heiner_abm.bench_inventory, heiner_abm.bench_bandit)
    Ref("adams2007", "Adams, R. P.; MacKay, D. J. C.", 2007, "Bayesian online changepoint detection", "preprint",
        journal="arXiv:0710.3742"),
    _a("fearnhead2007", "Fearnhead, P.; Liu, Z.", 2007, "On-line inference for multiple changepoint problems",
       "Journal of the Royal Statistical Society: Series B (Statistical Methodology)", "69", "4", "589–605"),
    _a("bental2013", "Ben-Tal, A.; den Hertog, D.; De Waegenaere, A.; Melenberg, B.; Rennen, G.", 2013,
       "Robust solutions of optimization problems affected by uncertain probabilities", "Management Science", "59",
       "2", "341–357"),
    _c("garivier2011", "Garivier, A.; Moulines, E.", 2011,
       "On upper-confidence bound policies for switching bandit problems",
       "Algorithmic Learning Theory: 22nd International Conference, ALT 2011 (Lecture Notes in Computer Science, "
       "Vol. 6925)", "J. Kivinen, C. Szepesvári, E. Ukkonen, & T. Zeugmann", "174–188", "Springer"),
    _a("auer2002", "Auer, P.; Cesa-Bianchi, N.; Fischer, P.", 2002,
       "Finite-time analysis of the multiarmed bandit problem", "Machine Learning", "47", "", "235–256"),
    _b("sutton2018", "Sutton, R. S.; Barto, A. G.", 2018, "Reinforcement learning: An introduction (2nd ed.)",
       "MIT Press"),
    _a("kirkpatrick1983", "Kirkpatrick, S.; Gelatt, C. D.; Vecchi, M. P.", 1983, "Optimization by simulated annealing",
       "Science", "220", "4598", "671–680"),
    # --- NK landscapes: interaction complexity
    _a("kauffman1987", "Kauffman, S.; Levin, S.", 1987, "Towards a general theory of adaptive walks on rugged landscapes",
       "Journal of Theoretical Biology", "128", "1", "11–45"),
    _b("kauffman1993", "Kauffman, S. A.", 1993, "The origins of order: Self-organization and selection in evolution",
       "Oxford University Press"),
    _a("levinthal1997", "Levinthal, D. A.", 1997, "Adaptation on rugged landscapes", "Management Science", "43", "7",
       "934–950"),
    _a("rivkin2000", "Rivkin, J. W.", 2000, "Imitation of complex strategies", "Management Science", "46", "6",
       "824–844"),
    # --- signal detection and methods
    _b("green1966", "Green, D. M.; Swets, J. A.", 1966, "Signal detection theory and psychophysics", "Wiley"),
    _a("swets1988", "Swets, J. A.", 1988, "Measuring the accuracy of diagnostic systems", "Science", "240", "4857",
       "1285–1293"),
    _a("hanley1982", "Hanley, J. A.; McNeil, B. J.", 1982,
       "The meaning and use of the area under a receiver operating characteristic (ROC) curve", "Radiology", "143",
       "1", "29–36"),
    _a("diebold1995", "Diebold, F. X.; Mariano, R. S.", 1995, "Comparing predictive accuracy",
       "Journal of Business & Economic Statistics", "13", "3", "253–263"),
    _a("chong1986", "Chong, Y. Y.; Hendry, D. F.", 1986, "Econometric evaluation of linear macro-economic models",
       "Review of Economic Studies", "53", "4", "671–690"),
    _a("stone1974", "Stone, M.", 1974, "Cross-validatory choice and assessment of statistical predictions",
       "Journal of the Royal Statistical Society: Series B", "36", "2", "111–147"),
    _b("efron1993", "Efron, B.; Tibshirani, R. J.", 1993, "An introduction to the bootstrap", "Chapman & Hall"),
    _a("hall1939", "Hall, R. L.; Hitch, C. J.", 1939, "Price theory and business behaviour", "Oxford Economic Papers",
       "2", "", "12–45"),
    _a("roth1995", "Roth, A. E.; Erev, I.", 1995,
       "Learning in extensive-form games: Experimental data and simple dynamic models in the intermediate term",
       "Games and Economic Behavior", "8", "1", "164–212"),
    _b("boyd1985", "Boyd, R.; Richerson, P. J.", 1985, "Culture and the evolutionary process",
       "University of Chicago Press"),
    _a("rockafellar2000", "Rockafellar, R. T.; Uryasev, S.", 2000, "Optimization of conditional value-at-risk",
       "Journal of Risk", "2", "3", "21–41"),
    _a("savage1951", "Savage, L. J.", 1951, "The theory of statistical decision",
       "Journal of the American Statistical Association", "46", "253", "55–67"),
    _b("axelrod1984", "Axelrod, R.", 1984, "The evolution of cooperation", "Basic Books"),
    _a("maynardsmith1973", "Maynard Smith, J.; Price, G. R.", 1973, "The logic of animal conflict", "Nature", "246",
       "5427", "15–18"),
    _a("nowak1993", "Nowak, M.; Sigmund, K.", 1993,
       "A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner's Dilemma game", "Nature",
       "364", "6432", "56–58"),
    _a("holm1979", "Holm, S.", 1979, "A simple sequentially rejective multiple test procedure",
       "Scandinavian Journal of Statistics", "6", "2", "65–70"),
    _a("mckay1979", "McKay, M. D.; Beckman, R. J.; Conover, W. J.", 1979,
       "A comparison of three methods for selecting values of input variables in the analysis of output from a "
       "computer code", "Technometrics", "21", "2", "239–245"),
    _a("bergstra2012", "Bergstra, J.; Bengio, Y.", 2012, "Random search for hyper-parameter optimization",
       "Journal of Machine Learning Research", "13", "", "281–305"),
    _b("saltelli2008", "Saltelli, A.; Ratto, M.; Andres, T.; Campolongo, F.; Cariboni, J.; Gatelli, D.; Saisana, M.; "
       "Tarantola, S.", 2008, "Global sensitivity analysis: The primer", "Wiley"),
    _a("nosek2018", "Nosek, B. A.; Ebersole, C. R.; DeHaven, A. C.; Mellor, D. T.", 2018, "The preregistration revolution",
       "Proceedings of the National Academy of Sciences", "115", "11", "2600–2606"),
    _c("tesfatsion2006", "Tesfatsion, L.", 2006, "Agent-based computational economics: A constructive approach to "
       "economic theory", "Handbook of computational economics (Vol. 2)", "L. Tesfatsion & K. L. Judd", "831–880",
       "Elsevier"),
    _a("davis2007", "Davis, J. P.; Eisenhardt, K. M.; Bingham, C. B.", 2007,
       "Developing theory through simulation methods", "Academy of Management Review", "32", "2", "480–499"),
    _a("harrison2007", "Harrison, J. R.; Lin, Z.; Carroll, G. R.; Carley, K. M.", 2007,
       "Simulation modeling in organizational and management research", "Academy of Management Review", "32", "4",
       "1229–1245"),
]
REFERENCES: Dict[str, Ref] = {r.key: r for r in _REFS}


def cite(*keys: str) -> str:
    """In-text citation list, e.g. cite('heiner1983', 'heiner1989') -> 'Heiner (1983); Heiner (1989)'."""
    return "; ".join(REFERENCES[k].cite for k in keys)


# ------------------------------------------------------------------------------------------------ hypotheses
Evidence = Tuple[str, str]      # (reference key, how the work bears on the hypothesis)


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    title: str
    where: str                           # page (and tab) that tests it
    rc_prediction: str
    alt_prediction: str
    alt_label: str                       # which rival theory the alternative comes from
    support: List[Evidence] = field(default_factory=list)
    alternative: List[Evidence] = field(default_factory=list)
    contribution: str = ""


HYPOTHESES: List[Hypothesis] = [
    Hypothesis(
        "H1", "Can less flexible firms outperform more flexible ones when flexibility is free?",
        "Hypothesis tests · H1",
        "When decision errors are costly, the profit–flexibility slope is negative even at zero cost of flexibility.",
        "Relaxing a constraint cannot make an optimizer worse off, so free flexibility never lowers profit.",
        "Neoclassical optimization",
        support=[("heiner1983", "Imperfect agents do better by restricting their repertoire when the reliability "
                                "condition fails; this holds without any cost of flexibility."),
                 ("bookstaber1985", "Under extended uncertainty, coarse rules that ignore information can be optimal."),
                 ("gigerenzer2009", "Less-is-more effects: simple heuristics beat flexible models when estimation "
                                    "error is large."),
                 ("thompson1967", "Organizations buffer their technical core from environmental fluctuation.")],
        alternative=[("samuelson1947", "Le Chatelier principle: removing a constraint weakly improves the optimum."),
                     ("stigler1939", "Flexible plants trade static efficiency for adaptability, and adaptability "
                                     "has value when conditions vary."),
                     ("teece1997", "Dynamic capabilities, the capacity to reconfigure, are a source of advantage."),
                     ("leonardbarton1992", "Core capabilities can harden into core rigidities that hamper "
                                           "adaptation.")],
        contribution="Tests the 'harmful free flexibility' claim in a strategic market where flexibility is a "
                     "behavioral parameter rather than a cost, and measures r, w, π, G and D for every decision "
                     "instead of inferring them."),
    Hypothesis(
        "H2", "Does the flexibility–profit relationship change sign as industry profitability rises?",
        "Hypothesis tests · H2",
        "Mistakes cost less in profitable markets, so the tolerance limit falls and flexibility becomes favored "
        "above a switch point.",
        "Firms below aspiration search and change more; poor performance triggers change rather than rigidity.",
        "Behavioral theory of the firm",
        support=[("heiner1983", "The tolerance limit (D/G)(1−π)/π rises when losses from mistakes are large "
                                "relative to gains."),
                 ("march1987", "Managers attend to survival; near a survival point they avoid risky departures."),
                 ("staw1981", "Threat produces rigid, well-learned responses (threat-rigidity).")],
        alternative=[("cyert1963", "Problemistic search: performance below aspiration triggers search for change."),
                     ("greve1998", "Performance below aspirations increases the likelihood of risky organizational "
                                   "change."),
                     ("kahneman1979", "Below a reference point decision makers become risk seeking.")],
        contribution="Separates what pays (the normative RC) from what firms do (descriptive aspiration theory) by "
                     "locating the profitability at which rigidity stops paying, under identical shocks."),
    Hypothesis(
        "H3", "How does environmental volatility affect the payoff to flexibility?",
        "Hypothesis tests · H3",
        "Volatility widens the gap between difficulty and competence (CD-gap), so reliability falls and the "
        "payoff to flexibility declines.",
        "Flexibility is an option whose value rises with volatility.",
        "Real options / value of flexibility",
        support=[("heiner1983", "Greater uncertainty (a wider CD-gap) makes rule-governed behavior optimal."),
                 ("makridakis2000", "In forecasting competitions simple methods perform as well as or better than "
                                    "complex ones on noisy series."),
                 ("eisenhardt2000", "In high-velocity markets effective capabilities are simple, experiential "
                                    "rules.")],
        alternative=[("stigler1939", "The value of productive flexibility grows with output variability."),
                     ("marschak1962", "Flexibility is valuable precisely because of uncertainty."),
                     ("dixit1994", "Option value increases with the volatility of the underlying."),
                     ("volberda1996", "Hypercompetitive environments require flexible organizational forms.")],
        contribution="Distinguishes volatility that raises decision difficulty from volatility that raises "
                     "option value by holding the decision rule fixed and measuring the CD-gap directly."),
    Hypothesis(
        "H4", "Do higher fixed costs change the profitability at which flexibility starts to pay?",
        "Hypothesis tests · H4",
        "Fixed costs shrink the net gains from good decisions and enlarge losses from bad ones, raising the "
        "tolerance limit and moving the switch point up.",
        "Fixed and sunk costs do not enter marginal decisions, so they cannot change which firms benefit from "
        "flexibility.",
        "Marginal analysis",
        support=[("heiner1983", "The tolerance limit depends on the stakes D/G, not only on reliability."),
                 ("march1987", "Proximity to a survival threshold changes attention and risk taking."),
                 ("kahneman1979", "Outcomes are evaluated relative to a reference point; losses loom larger.")],
        alternative=[("samuelson1947", "Optimal choices depend only on marginal conditions."),
                     ("stigler1939", "Short-run production decisions are governed by marginal cost.")],
        contribution="Provides a controlled test in which fixed costs change decision stakes but not marginal "
                     "costs, with a built-in control condition in which decisions ignore fixed costs."),
    Hypothesis(
        "H5", "Does competition intensity change the profit level at which flexibility is favored?",
        "Hypothesis tests · H5",
        "Exploratory: competition changes the stakes of each decision, shifting the switch point.",
        "Competition intensity changes the level of profits but not the ranking of flexible versus rigid firms.",
        "No specific rival prediction",
        support=[("nickell1996", "Competition is associated with faster productivity growth: it changes the "
                                 "returns to adaptive effort."),
                 ("aghion2005", "The effect of competition on adaptive investment is non-monotone (inverted U).")],
        alternative=[("stigler1964", "Oligopoly outcomes are shaped by the difficulty of coordination rather than "
                                     "by firms' adjustment behavior."),
                     ("vegaredondo1997", "Imitative adjustment drives oligopolies toward the competitive outcome "
                                         "regardless of initial behavior.")],
        contribution="Exploratory: maps how the switch point moves with competition intensity, a relationship the "
                     "reliability literature has not formalized."),
    Hypothesis(
        "H6", "Does industry-wide rigidity keep markets away from equilibrium, and profitable, without collusion?",
        "Hypothesis tests · H6",
        "When errors are punishing, rigid behavior is individually rational and raises industry profit; no "
        "coordination is needed.",
        "Margins above the competitive level require (tacit) collusion; learning drives markets to equilibrium.",
        "Oligopoly / collusion theory",
        support=[("heiner1983", "Predictable, rule-governed behavior arises from uncertainty, not agreement."),
                 ("nerlove1958", "Slow (adaptive) adjustment stabilizes cobweb markets."),
                 ("heiner1988b", "Delaying adjustment can be necessary for imperfect agents."),
                 ("carlson1968", "Cobweb markets with suitably formed price expectations are stable."),
                 ("carlton1986", "Many transaction prices are rigid for long periods."),
                 ("sonnemans2004", "The mix of expectation rules agents use shapes whether a cobweb "
                                   "economy converges."),
                 ("hommes2007", "In cobweb experiments, prices do not always converge to the "
                                "rational-expectations equilibrium.")],
        alternative=[("stigler1964", "Supra-competitive margins require detecting and punishing deviations."),
                     ("green1984", "Collusion can be sustained under imperfect price information."),
                     ("vegaredondo1997", "Imitation of the most successful firm leads to Walrasian outcomes."),
                     ("huck1999", "Experimental Cournot markets with imitation become more competitive.")],
        contribution="Shows how rigidity chosen for reliability reasons can produce collusion-like margins without "
                     "any coordination, a non-collusive account of sticky, profitable oligopoly."),
    Hypothesis(
        "H7", "Does competence change the value of flexibility?",
        "Hypothesis tests · H7",
        "Better foresight narrows the CD-gap: r rises, w falls, and the payoff to flexibility rises.",
        "Less uncertainty lowers the option value of flexibility, so its payoff falls as foresight improves.",
        "Real options",
        support=[("heiner1983", "Competence is one side of the CD-gap; raising it restores the reliability of "
                                "deviations."),
                 ("jones1984", "Flexibility is more valuable when more information is expected before commitment."),
                 ("muth1960", "The optimal weight on new information rises with its signal-to-noise ratio.")],
        alternative=[("dixit1994", "Resolving uncertainty lowers the value of keeping options open."),
                     ("kreps1979", "Preference for flexibility stems from uncertainty about future contingencies.")],
        contribution="Crosses competence with difficulty in one design, separating the competence side of the "
                     "CD-gap from environmental uncertainty."),
    Hypothesis(
        "H8", "How do perception errors affect the value of flexibility?",
        "Hypothesis tests · H8",
        "Noisy perception lowers competence, so the payoff to flexibility falls with noise.",
        "Noise can act as exploration, and uncertainty raises option value, so flexibility may gain.",
        "Real options / exploration",
        support=[("heiner1983", "Perception errors reduce reliability r/w."),
                 ("kalman1960", "Optimal filters put less weight on noisier signals."),
                 ("brainard1967", "Uncertainty about effects calls for attenuated responses."),
                 ("geman1992", "Estimation variance penalizes flexible models.")],
        alternative=[("dixit1994", "Greater uncertainty raises option value."),
                     ("march1991", "Variability supports exploration, which can improve long-run adaptation."),
                     ("kirkpatrick1983", "Random perturbation helps a search process escape poor local states.")],
        contribution="Tests whether agent-side noise and environment-side volatility have the same effect; they "
                     "are often conflated as 'uncertainty'."),
    Hypothesis(
        "H9", "Under rising uncertainty, which selection rules win: always adjusting, or deviating rarely or only on clear signals?",
        "Hypothesis tests · H9",
        "Selective rules (act only on large, clear signals) and the default rule gain on the always-adjust rule "
        "as volatility rises.",
        "Incremental small adjustments are safest; or, without adjustment costs, always adjusting is optimal.",
        "Incrementalism / frictionless optimization",
        support=[("heiner1986", "Reliable action requires acting only on signals that discriminate well (hit rate "
                                "versus false-alarm rate)."),
                 ("green1966", "The optimal detection criterion rises when false alarms are costly or signals weak."),
                 ("scarf1960", "(S, s) policies: act only when the gap is large."),
                 ("caballero1999", "Adjustment is lumpy: firms act on large imbalances.")],
        alternative=[("lindblom1959", "Successive limited comparisons: small incremental changes limit the cost of "
                                      "error."),
                     ("sheshinski1977", "Without adjustment costs the inaction band vanishes and continuous "
                                        "adjustment is optimal.")],
        contribution="Compares selection rules head to head on identical shocks, linking the RC to "
                     "signal-detection criteria and to (S, s) inaction bands."),
    Hypothesis(
        "H10", "Does greater uncertainty make behavior more or less predictable?",
        "Hypothesis tests · H10",
        "Agents that learn when deviating pays deviate less often as uncertainty rises: behavior becomes more "
        "rule-governed.",
        "Performance shortfalls trigger search, so volatility increases change; or optimizers re-optimize more "
        "often when shocks are larger.",
        "Satisficing / optimization",
        support=[("heiner1983", "The origin of predictable behavior is uncertainty about which action is best."),
                 ("nelson1982", "Firms operate through routines that change slowly."),
                 ("friedman1960", "Rules outperform discretion when effects are uncertain and lagged."),
                 ("dixit1989", "The zone of inaction widens with uncertainty.")],
        alternative=[("cyert1963", "Search is triggered by performance below aspiration."),
                     ("greve1998", "Shortfalls increase risky change."),
                     ("kydland1977", "Rules can arise from commitment problems rather than from uncertainty.")],
        contribution="Endogenizes rule-following: agents learn from counterfactual payoffs when to deviate, so "
                     "predictability emerges rather than being imposed."),
    Hypothesis(
        "H11", "Does the number of rivals change the value of flexibility?",
        "Hypothesis tests · H11",
        "A target that moves with more rivals is harder to hit, so the reliable adjustment speed and the payoff "
        "to flexibility fall with n.",
        "Only dynamic stability matters: flexibility hurts when adjustment exceeds the stability limit, "
        "regardless of errors.",
        "Cobweb stability theory",
        support=[("heiner1989", "The maximal reliable adjustment speed falls as the slope of the target map "
                                "steepens."),
                 ("huck2004", "Behavior in experimental oligopolies changes qualitatively with the number of "
                              "firms.")],
        alternative=[("theocharis1960", "Full best-reply Cournot adjustment is not stable with three or "
                                        "more firms."),
                     ("vegaredondo1997", "Imitation drives any number of firms toward the competitive output.")],
        contribution="Separates error-driven from stability-driven harm of flexibility by measuring the "
                     "error-to-signal ratio alongside the stability radius."),
    Hypothesis(
        "H12", "Does slower learning of a structural change affect how flexibility pays for model-based firms?",
        "Hypothesis tests · H12",
        "A persistent misspecification (unannounced structural change) widens the CD-gap, so the payoff to flexibility "
        "falls with the model-updating lag.",
        "Unforeseen contingencies raise the value of flexibility.",
        "Preference for flexibility",
        support=[("knight1921", "Uncertainty that cannot be reduced to known probabilities."),
                 ("hansen2008", "Decision makers who fear model misspecification choose more cautious policies."),
                 ("gilboa1989", "Ambiguity leads to cautious (maxmin) choice.")],
        alternative=[("kreps1979", "Uncertainty about future preferences or contingencies creates a preference for "
                                   "flexibility."),
                     ("teece1997", "Rapid, unforeseen change rewards reconfiguration capabilities.")],
        contribution="Introduces model misspecification (unannounced demand-regime shifts) as a distinct source of "
                     "the CD-gap and tests it against risk at matched unpredictability."),
    Hypothesis(
        "RC", "Does the reliability condition predict, out of sample, which firms benefit from flexibility?",
        "Heiner: does the RC predict performance?",
        "Firms with a larger RC margin, estimated in one window, beat their rigid twin more often in the next "
        "window (AUC > 0.5).",
        "Flexibility always pays (a constant forecast, AUC = 0.5), or past performance alone predicts future "
        "performance.",
        "Optimization / reinforcement learning",
        support=[("heiner1986", "r and w are hit and false-alarm rates, so the RC can be scored like a detector."),
                 ("hanley1982", "The AUC measures discrimination independently of base rates."),
                 ("stone1974", "Out-of-sample validation guards against overfitting.")],
        alternative=[("erev1998", "Simple reinforcement learning predicts behavior and payoffs well."),
                     ("sutton2018", "Value estimates learned from experience guide action without a structural "
                                    "model.")],
        contribution="Turns the RC from an ex-post explanation into an ex-ante forecast, scored out of sample with "
                     "a rigid-twin counterfactual for each firm."),
    Hypothesis(
        "DRC", "Does a one-period valuation misjudge decisions that persist and provoke reactions?",
        "Heiner: dynamic RC (1989) · ①",
        "The value of a decision includes persistence and strategic feedback; the one-period RC omits them.",
        "Decisions are made at the margin period by period, so the immediate effect is sufficient.",
        "Myopic marginal analysis",
        support=[("heiner1989", "Imperfect agents facing moving targets must be analyzed dynamically."),
                 ("levinthal1993", "Learning is myopic: it overweights near-term and local consequences.")],
        alternative=[("muth1961", "With rational expectations, period-by-period optimization is consistent."),
                     ("ezekiel1938", "Cobweb dynamics are driven by period-by-period supply decisions.")],
        contribution="Decomposes a decision's value into immediate, persistence and strategic-feedback components, "
                     "extending the RC from a one-shot bet to a dynamic setting."),
    Hypothesis(
        "BOUND", "Where does the profit-maximizing adjustment speed lie relative to Heiner's reliability bound?",
        "Heiner: dynamic RC (1989) · ②",
        "The best partial-adjustment speed is at or below β₀ = 1/((1+K)(1−f′)) and falls as the error-to-signal "
        "ratio K rises.",
        "Full adjustment to the best reply is optimal; slower adjustment only matters for stability.",
        "Best-reply dynamics / cobweb stability",
        support=[("heiner1989", "Theorem 2 bounds the speed that converges despite decision errors."),
                 ("muth1960", "Optimal smoothing weights fall as noise rises."),
                 ("brainard1967", "Parameter uncertainty attenuates optimal responses.")],
        alternative=[("theocharis1960", "Stability depends on the number of firms and adjustment speed only."),
                     ("nerlove1958", "Adaptive adjustment coefficients are set by stability considerations.")],
        contribution="Tests Heiner's dynamic bound quantitatively, with K measured from the agents' own decision "
                     "errors."),
    Hypothesis(
        "SDT", "How do discriminability and the best decision threshold change as the CD-gap widens?",
        "Heiner: dynamic RC (1989) · ③",
        "The ROC area of the deviation signal falls with uncertainty; the value-maximizing threshold rises.",
        "A perfectly competent agent (AUC = 1) should always act on exceptions; thresholds reflect only "
        "adjustment costs.",
        "Frictionless optimization / (S, s) theory",
        support=[("heiner1986", "The RC is a signal-detection criterion."),
                 ("green1966", "The optimal criterion depends on base rates and payoffs."),
                 ("swets1988", "ROC analysis separates discrimination from response bias.")],
        alternative=[("scarf1960", "Inaction bands arise from fixed adjustment costs."),
                     ("caballero1999", "Lumpy adjustment reflects adjustment costs, not perception.")],
        contribution="Derives each firm's ROC curve from its own decisions and shows the tolerance limit acting as "
                     "the optimal detection criterion."),
    Hypothesis(
        "KNIGHT", "Does unannounced structural change affect model-based flexibility differently from risk?",
        "Risk vs structural change · ①",
        "At matched unpredictability, misspecified models make flexibility harmful for model-based firms but not "
        "for model-free firms.",
        "Uncertainty of either kind raises the option value of flexibility; or no theory can be formulated for "
        "genuine uncertainty.",
        "Real options / Knight–Lucas skepticism",
        support=[("knight1921", "Risk and uncertainty are different in kind."),
                 ("hansen2001", "Model uncertainty calls for robust rather than fine-tuned decisions."),
                 ("heiner1983", "The RC applies whatever the source of the CD-gap.")],
        alternative=[("lucas1977", "In cases of genuine uncertainty economic reasoning has limited value."),
                     ("arrow1951", "Choice theory under uncertainty needs known probability structures."),
                     ("kreps1979", "Unforeseen contingencies raise the value of flexibility.")],
        contribution="Compares risk and structural uncertainty at matched unpredictability in the same market, "
                     "for model-based and model-free decision rules."),
    Hypothesis(
        "PUNCT", "Is adjustment around structural shifts gradual or punctuated?",
        "Risk vs structural change · ②",
        "Imperfect agents keep adjusting slowly until they can locate the new equilibrium reliably, then jump.",
        "Adjustment is smooth and proportional to the shift (adaptive expectations).",
        "Adaptive expectations",
        support=[("heiner1989", "Punctuated adjustment around shifting equilibria."),
                 ("caballero1999", "Lumpy adjustment at the micro level.")],
        alternative=[("nerlove1958", "Adaptive expectations imply smooth geometric adjustment."),
                     ("muth1960", "Exponential smoothing adjusts by a constant share of each surprise.")],
        contribution="Provides an event-study test of punctuated adjustment around unannounced regime shifts."),
    Hypothesis(
        "CDGAP", "How does reliability change as difficulty rises relative to competence?",
        "Heiner: CD-gap explorer",
        "r(U) falls and w(U) rises with the CD-gap; raising competence restores them.",
        "Information and flexibility are both valuable, with no systematic trade-off between them.",
        "Value of information",
        support=[("heiner1983", "Defines the CD-gap and its effect on r and w."),
                 ("heiner1988", "Imperfect decisions are unavoidable when competence falls short of difficulty."),
                 ("simon1956", "Rational behavior is shaped jointly by the environment and the agent's "
                               "capacities.")],
        alternative=[("marschak1962", "Better information raises the value of flexibility."),
                     ("jones1984", "Information and flexibility are complements.")],
        contribution="Measures r and w on a grid of difficulty × competence, making the CD-gap an observable rather "
                     "than a construct."),
    Hypothesis(
        "EVO", "Which flexibility do industries evolve toward as uncertainty rises?",
        "Endogenous flexibility",
        "Through imitation of successful rivals, evolved flexibility falls as volatility rises.",
        "Selection favors flexibility where it is most valuable (high volatility); or inertia is selected "
        "everywhere.",
        "Evolutionary economics / organizational ecology",
        support=[("heiner1983", "Rule-governed behavior is selected under greater uncertainty."),
                 ("alchian1950", "Selection among firms can produce adaptive behavior without optimization.")],
        alternative=[("hannan1984", "Selection favors structural inertia in any environment."),
                     ("vegaredondo1997", "Imitation of the best leads to aggressive, competitive behavior."),
                     ("brock1997", "Agents switch toward costly sophisticated rules when they pay, which can "
                                   "destabilize markets.")],
        contribution="Lets flexibility evolve by social learning, testing whether reliability considerations "
                     "shape industry structure without optimization."),
]
HYPOTHESES.append(Hypothesis(
    "ARENA", "When rival theories compete as agents, which earn most and which resist invasion?",
    "Agent tournament",
    "An agent that deviates from rule B only when its learned reliability condition holds out-earns rival decision "
    "rules in mixed markets, and a population of such agents cannot be invaded.",
    "The best-performing rule is the one that best fits the market's structure: model-free adaptation, imitation or "
    "optimization, depending on the environment.",
    "Rival decision rules",
    support=[("heiner1983", "Rule-governed behavior that deviates only when reliable should outperform under "
                            "uncertainty."),
             ("heiner1989", "Partial, reliability-limited adjustment toward a moving target outperforms full "
                            "adjustment."),
             ("axelrod1984", "Strategy tournaments reveal which decision rules do well against a field of rivals.")],
    alternative=[("nerlove1958", "Adaptive expectations based on observed prices track a cobweb market without a "
                                 "structural model."),
                 ("vegaredondo1997", "Imitating the most profitable firm earns more than rivals in Cournot markets."),
                 ("erev1998", "Simple reinforcement learning predicts payoffs in repeated games."),
                 ("kalman1960", "Optimal filtering yields the best forecasts for a model-based optimizer.")],
    contribution="Implements each rival theory as an agent with an equal tuning budget and pre-registered, held-out "
                 "evaluation, so theories compete on behavior rather than on predictions written by the modeler."))
HYPOTHESES.append(Hypothesis(
    "MECH1", "Does restricting an unreliable flexible rule pay, and what does learning its reliability cost?",
    "Heiner: mechanisms · ①",
    "With known reliability, restricting deviations beats always adjusting toward an error-prone target; an agent "
    "that must estimate its own reliability loses part or all of that gain, through limited experience and through "
    "judging its past decisions with a misspecified model.",
    "Always adjusting toward the best available target is optimal (no restriction needed), or learned estimates are "
    "as good as true ones, so the principle and its implementation coincide.",
    "Optimization / adaptive learning",
    support=[("heiner1983", "Restriction to rule B pays when the reliability ratio falls short of the tolerance "
                            "limit."),
             ("conlisk1996", "Deciding how to decide is itself costly and can regress without end."),
             ("hansen2008", "Judging policies with a misspecified model biases decisions.")],
    alternative=[("muth1960", "Optimal smoothing absorbs noise without a separate selection rule."),
                 ("sutton2018", "Learned value estimates converge to true values with enough experience.")],
    contribution="Separates the reliability principle from the cost of applying it, and splits that cost into "
                 "estimation from limited experience and judging with a misspecified model, using an oracle the "
                 "simulator can provide."))
HYPOTHESES.append(Hypothesis(
    "MECH2", "Where does restricting flexibility pay, and does the type of uncertainty matter?",
    "Heiner: mechanisms · ②",
    "The gain from reliability-based restriction rises with the measured error-to-signal ratio K of the flexible rule, "
    "and model misspecification raises it more than cost volatility (risk) does.",
    "Uncertainty of any kind raises the value of flexibility, or only dynamic stability matters.",
    "Real options / cobweb stability",
    support=[("heiner1989", "The reliable adjustment speed falls as the error-to-signal ratio K rises."),
             ("knight1921", "Uncertainty that cannot be reduced to known probabilities differs in kind from risk."),
             ("gigerenzer2009", "Which rule works depends on the structure of the environment (ecological "
                                "rationality).")],
    alternative=[("dixit1994", "Option value increases with volatility."),
                 ("theocharis1960", "Adjustment dynamics depend on market structure, not on the source of noise.")],
    contribution="Maps when each decision rule works across separate sources of uncertainty, and tests the "
                 "reliability condition as a boundary prediction against the measured reliability of the flexible "
                 "rule."))
HYPOTHESES.append(Hypothesis(
    "EMERGE", "Which decision rules emerge when firms choose their own rules?",
    "Rule choice",
    "When firms switch between rules according to recent performance, populations drift toward restricted, "
    "rule-governed rules (and change output less often) as uncertainty rises, and such markets stay further from "
    "equilibrium.",
    "Performance-based selection favors the most responsive, best-informed rule (or drives the market toward the "
    "equilibrium), whatever the level of uncertainty.",
    "Rational expectations / evolutionary selection",
    support=[("heiner1983", "Predictable, rule-governed behavior originates in uncertainty about which actions "
                            "are best."),
             ("brock1997", "With performance-based switching between cheap simple and costly sophisticated "
                           "predictors, simple rules can prevail and markets fluctuate around the equilibrium."),
             ("anufriev2012", "In the laboratory, simple forecasting heuristics compete by past performance and "
                              "their mix determines whether markets converge or oscillate.")],
    alternative=[("muth1961", "Expectations converge on the model's prediction, so the most informed rule wins."),
                 ("alchian1950", "Selection favors whichever behavior earns most, not restriction as such."),
                 ("vegaredondo1997", "Imitation of success drives oligopolies to the competitive outcome.")],
    contribution="Turns Heiner's claim about the origin of predictable behavior into an emergent outcome: rules are "
                 "not imposed but chosen by performance (logit switching with an adjustable intensity of choice), "
                 "and the resulting rule mix is linked to price volatility and the distance from equilibrium."))
HYPOTHESES.append(Hypothesis(
    "PATTERN", "Does the market reproduce documented field patterns?",
    "Field patterns",
    "A market of boundedly rational firms reproduces several independent empirical patterns at once: cobweb cycles, "
    "damping by adaptive adjustment, sticky and lumpy adjustment, imitation beyond Cournot–Nash, excess volatility "
    "around equilibrium and positive markups.",
    "Under rational expectations and flexible adjustment there are no systematic cycles and no stickiness, so these "
    "patterns would reflect frictions outside the model.",
    "Rational expectations / frictionless adjustment",
    support=[("grimm2005", "Matching several patterns at once constrains agent-based models far more than fitting "
                           "a single one."),
             ("ezekiel1938", "Lagged supply response produces alternating price cycles."),
             ("carlton1986", "Many transaction prices stay unchanged for long spells and then move in steps.")],
    alternative=[("muth1961", "Rational expectations remove systematic, predictable price cycles."),
                 ("samuelson1947", "Comparative statics treats markets as at equilibrium, adjusting smoothly.")],
    contribution="Pattern-oriented validation: the market is judged by whether it reproduces documented patterns "
                 "with criteria fixed in advance, so results cannot be dismissed as mere properties of the model."))
HYPOTHESES.append(Hypothesis(
    "CALIB", "Which decision rules predict laboratory behavior out of sample?",
    "Calibration",
    "Fitted to laboratory cobweb and Cournot data, simple rules, including rules that keep the previous decision "
    "unless a condition is met, predict subjects' later choices better than the rational benchmark, and subjects "
    "are heterogeneous in which rule fits best.",
    "Subjects' choices are best predicted by rational expectations or best replies; deviations are noise.",
    "Rational expectations / best reply",
    support=[("hommes2011", "Heterogeneous simple heuristics explain laboratory expectations better than a single "
                            "rational rule."),
             ("huck1999", "In Cournot experiments, imitation and best-reply rules both describe behavior, depending "
                          "on the information given."),
             ("heiner1983", "Behavior that keeps to a narrow repertoire is predicted to be more regular than "
                            "optimization implies.")],
    alternative=[("muth1961", "Expectations are model-consistent on average."),
                 ("theocharis1960", "Best-reply dynamics describe how Cournot markets adjust."),
                 ("hommes2007", "In stable cobweb treatments, average expectations come close to rational.")],
    contribution="Fits every theory's decision rule to the same subjects and compares them out of sample (first half "
                 "fitted, second half predicted), rather than in sample."))
HYPOTHESES.append(Hypothesis(
    "EXPER", "Which sources of uncertainty change how people adjust, and does a reliability aid change the outcome?",
    "Play the market · Human experiments: analysis",
    "Separating the sources of uncertainty, Heiner's account predicts less frequent adjustment where the information "
    "behind a change is less reliable (noisier cost estimates, unannounced demand shifts) and that a reliability-based "
    "aid improves relative profit; the effect of adjustment frequency on profit itself is not identified by the design "
    "and is reported as an association.",
    "People adjust more when conditions change more (volatility and shifts raise adjustment), responsiveness pays, "
    "and an aid adds nothing beyond attention; or adjustment reflects noise, inertia and anchoring unrelated to "
    "reliability.",
    "Reinforcement learning / behavioral bias",
    support=[("heiner1983", "Greater uncertainty narrows the repertoire of actions, making behavior more "
                            "predictable."),
             ("bolton2008", "Restricting how often newsvendor orders may change improves performance in the "
                            "laboratory."),
             ("fischbacher2007", "Computerized market experiments with a fixed protocol make choices comparable "
                                 "across participants.")],
    alternative=[("erev1998", "Reinforcement learning explains experimental behavior without any restriction "
                              "rule."),
                 ("schweitzer2000", "Newsvendor decisions are pulled toward mean demand and chase recent demand: "
                                    "a bias, not a reliability-based restriction."),
                 ("kahneman1979", "Choices under risk reflect reference dependence rather than reliability.")],
    contribution="Puts human participants in the same market as the agents, varies volatility, observation noise "
                 "and regime change one at a time within subjects, randomizes a reliability-gated decision aid "
                 "against an unaided control, and models adjustment probability and magnitude separately with "
                 "held-out prediction and recovery checks."))
HYPOTHESES.append(Hypothesis(
    "GEN", "Does the boundary between reliable and unreliable flexibility hold beyond the market?",
    "Generalization",
    "In other decision tasks with a default, a flexible alternative and a gap between difficulty and competence "
    "(an inventory task, a learning task with shifting payoffs and an irreversible investment task), restriction pays "
    "only where the flexible rule is "
    "unreliable, with the same error-to-signal boundary as in the market.",
    "The optimal policy in each task uses all available information (an updated order level, a learner that tracks "
    "volatility), so restricting the flexible rule never pays where it is used optimally.",
    "Optimal inventory / Bayesian learning",
    support=[("heiner1983", "The reliability condition is stated for any decision with a default and possible "
                            "deviations, not for markets in particular."),
             ("heiner1989", "The reliable adjustment speed falls as the error-to-signal ratio rises."),
             ("gigerenzer2009", "Simple rules win where estimation error is large relative to what can be "
                                "learned.")],
    alternative=[("arrow1951b", "Optimal inventory policies follow from the demand distribution."),
                 ("scarf1960", "Inaction in inventory policy arises from fixed ordering costs, not reliability."),
                 ("behrens2007", "People raise their learning rate when the environment is volatile, as an optimal "
                                 "learner does.")],
    contribution="Shows that the finding is about decision making, not about cobweb markets: the same selection "
                 "layers and the same error-to-signal boundary are tested in tasks with a different payoff "
                 "structure."))
HYPOTHESES.append(Hypothesis(
    "TRACK", "Do lopsided stakes call for restriction even with an optimal filter?",
    "Heiner vs optimal filtering",
    "Holding the information (signal-to-noise ratio) fixed, lopsided stakes (a wrong move costs more than a right move "
    "gains) make it pay to restrict the filter's moves in the costly direction, the more so the more lopsided the "
    "stakes.",
    "The best response to news depends only on the signal-to-noise ratio (the Kalman gain); stakes shift the level of "
    "the action but not how much it responds to new information (certainty equivalence).",
    "Optimal filtering",
    support=[("heiner1983", "The tolerance limit for acting on a signal rises with the loss from a wrong action "
                            "relative to the gain from a right one."),
             ("heiner1989", "The reliable adjustment speed depends on the error-to-signal ratio and on the costs of "
                            "errors."),
             ("bookstaber1985", "Under uncertainty, coarse rules that ignore some information can be optimal.")],
    alternative=[("muth1960", "Exponential smoothing with the signal-to-noise-optimal weight is the optimal forecast "
                              "for a random walk observed with noise."),
                 ("kalman1960", "The optimal filter's gain depends only on the noise variances."),
                 ("brainard1967", "With additive uncertainty, certainty equivalence holds and responses are not "
                                  "attenuated.")],
    contribution="Pins down which prediction is uniquely Heiner's in a case with an exact solution: the simulation "
                 "first reproduces the Muth–Kalman optimum, then shows that lopsided stakes, not information, drive "
                 "restriction, and that the restriction becomes unnecessary once the stakes are built into the "
                 "estimate."))
HYPOTHESES.append(Hypothesis(
    "EMPVAL", "Which decision rules describe human choices in public experimental data?",
    "Empirical validation",
    "In public Cournot, learning-to-forecast and newsvendor data, rules that keep the previous decision unless a "
    "deviation has proved reliable predict participants' later choices at least as well as the flexible rules they "
    "restrict; behavior is heterogeneous; and markets simulated with the fitted rules reproduce how often people "
    "change and how competitive the markets become.",
    "Human choices are best described by best replies, adaptive learning or systematic biases (anchoring, "
    "pull-to-center); restricting adjustment adds nothing out of sample.",
    "Adaptive learning / behavioral bias",
    support=[("hommes2011", "Heterogeneous simple heuristics describe laboratory expectations better than a single "
                            "rational rule."),
             ("hommes2005", "Participants coordinate on common, simple forecasting rules."),
             ("anufriev2012", "Participants switch between simple heuristics according to their past performance."),
             ("gomezmartinez2016", "Information about individual competitors makes Cournot markets more competitive, "
                                   "consistent with imitation and learning from rivals.")],
    alternative=[("bostian2008", "Adaptive learning explains the newsvendor pull-to-center effect."),
                 ("brokesova2022", "Pull-to-center reflects structural features of the task (prospect theory, "
                                   "impulse balance), not a domain-specific rule."),
                 ("evans2025", "A unified model of adaptive learning, level-k reasoning and replicator dynamics "
                               "predicts individual and market responses to announced changes."),
                 ("hommes2008", "Trend-following expectations generate bubbles in asset markets.")],
    contribution="Confronts the simulation's decision rules with public experimental data under a protocol fixed in "
                 "advance (earlier periods to fit, later periods to test, treatment-specific information sets, "
                 "participant-level uncertainty), and adds a generative check: markets re-run with the fitted rules "
                 "must reproduce the human markets."))
HYPOTHESIS_BY_ID: Dict[str, Hypothesis] = {h.hid: h for h in HYPOTHESES}


# ------------------------------------------------------------------------------------------------ rival theories
THEORY_SOURCES: Dict[str, List[str]] = {
    "heiner": ["heiner1983", "heiner1985", "heiner1986", "heiner1988", "heiner1989"],
    "neo": ["samuelson1947", "muth1961", "stigler1939"],
    "options": ["stigler1939", "marschak1962", "kreps1979", "dixit1989", "dixit1994"],
    "cobweb": ["ezekiel1938", "nerlove1958", "theocharis1960", "carlson1967", "hommes1994"],
    "biasvar": ["geman1992", "gigerenzer1999", "gigerenzer2009", "dawes1979"],
    "satisficing": ["simon1955", "cyert1963", "greve1998"],
    "rl": ["roth1995", "erev1998", "sutton2018"],
    "imitation": ["alchian1950", "nelson1982", "vegaredondo1997"],
    "ecology": ["hannan1977", "hannan1984", "amburgey1993"],
}


# ------------------------------------------------------------------------------------------------ contribution
CONTRIBUTIONS: List[Tuple[str, str, List[str]]] = [
    ("Decision-level measurement of the reliability condition",
     "Every decision is classified as a correct deviation, a missed exception (type I) or a wrong deviation (type "
     "II), so π, r, w, G and D are measured per firm rather than assumed. Direct empirical tests of the RC remain "
     "scarce; agent-based simulation makes its quantities observable.",
     ["heiner1983", "heiner1986", "conlisk1996", "tesfatsion2006"]),
    ("A dynamic reliability condition",
     "The RC values each deviation as a one-shot bet. Decisions in markets persist and provoke reactions, so the "
     "model values them over a horizon, with and without rival reactions, and decomposes the difference.",
     ["heiner1989", "levinthal1993"]),
    ("Out-of-sample forecasting tests",
     "The RC is estimated in one window and used to predict, in the next, whether each firm beats its own rigid "
     "twin. This moves the theory from post-hoc explanation to prediction, scored with the AUC.",
     ["hanley1982", "stone1974", "efron1993"]),
    ("Head-to-head adjudication of rival theories",
     "Heiner's theory, optimization, real options, cobweb stability, bias–variance reasoning, satisficing and "
     "structural inertia are confronted with the same discriminating experiments, and their forecasts are compared "
     "with encompassing tests.",
     ["chong1986", "diebold1995", "davis2007"]),
    ("A fair agent tournament",
     "Each theory is implemented as a decision rule and competes in the same market. Every rule gets the same "
     "tuning budget on training environments; evaluation uses held-out environments, a frozen and hashed analysis "
     "plan, head-to-head profit, invasion tests, global sensitivity analysis and replication across seeds.",
     ["axelrod1984", "maynardsmith1973", "bergstra2012", "nosek2018", "saltelli2008"]),
    ("When does restricting flexibility pay, and who should decide?",
     "The tournament separates the principle from its implementation. Restricting a flexible rule pays when that rule "
     "is unreliable (a misspecified model) and costs when it is reliable, as the reliability condition implies. But an "
     "agent that estimates its own reliability case by case faces a second-order inference problem, and fixed rules "
     "capture the gains more robustly. This connects the reliability condition to the regress problem of deciding "
     "how to decide.",
     ["heiner1983", "conlisk1996", "hansen2008"]),
    ("Risk versus unannounced structural change",
     "Structural change that agents are not told about (misspecified demand models) is compared with risk at "
     "matched unpredictability, testing whether the RC applies whatever the source of the CD-gap. The shifts are "
     "specified by the simulator, so this is not Knightian uncertainty in the unrestricted sense.",
     ["knight1921", "hansen2008", "lucas1977"]),
    ("Which predictions are unique to Heiner",
     "Several predictions are shared with optimal filtering, signal detection or bias–variance reasoning (for "
     "example, slower adjustment when noise rises). The predictions unique to the RC are that the stakes (the "
     "tolerance limit) matter beyond accuracy, and that free flexibility can lower profit. The design isolates "
     "them.",
     ["muth1960", "kalman1960", "green1966", "gigerenzer2009"]),
]

METHOD_REFS = ["creed2021", "holm1979", "mckay1979", "nowak1993", "hall1939", "roth1995", "boyd1985", "rockafellar2000",
               "savage1951", "tesfatsion2006", "davis2007", "harrison2007", "hanley1982", "stone1974", "efron1993",
               "diebold1995", "chong1986", "adams2007", "fearnhead2007", "bental2013", "garivier2011", "auer2002",
               "kauffman1987", "kauffman1993", "levinthal1997", "rivkin2000"]


def cited_keys() -> List[str]:
    """Every reference key cited anywhere in the registry (for integrity checks and the bibliography)."""
    keys = set()
    for h in HYPOTHESES:
        keys.update(k for k, _ in h.support + h.alternative)
    for v in THEORY_SOURCES.values():
        keys.update(v)
    for _, _, ks in CONTRIBUTIONS:
        keys.update(ks)
    keys.update(METHOD_REFS)
    return sorted(keys)


def bibliography(keys: Optional[List[str]] = None) -> List[Ref]:
    refs = [REFERENCES[k] for k in (keys or REFERENCES)]
    return sorted(refs, key=lambda r: (r.authors.lower(), r.year))
