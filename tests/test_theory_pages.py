"""Theory pages: every theory is complete and consistent with the agents, the registry and the registered runs."""
from heiner_abm.arena import KEYS, PREREG, THEORY_DESIGNS
from heiner_abm.literature import HYPOTHESIS_BY_ID, REFERENCES
from heiner_abm.mechanisms import default_plan
from heiner_abm.registered import DIRECTIONAL, STUDY_PLAN, TOURNAMENT, TOURNAMENT_PLAN
from heiner_abm.theories import THEORIES as DIRECTIONAL_THEORIES
from heiner_abm.theory_content import THEORIES
from ui.illustrations import ILLUSTRATIONS


def test_every_tournament_theory_has_a_page_and_illustration():
    assert {t.key for t in THEORIES} == set(KEYS) - {"ruleb"}
    assert set(ILLUSTRATIONS) == {t.key for t in THEORIES}
    for t in THEORIES:
        assert t.key in THEORY_DESIGNS and len(t.strengths) >= 2 and len(t.limits) >= 2
        assert t.origins and t.formal and t.flexibility and t.page.startswith("app_pages/theory_")


def test_references_and_hypotheses_resolve():
    for t in THEORIES:
        assert all(k in REFERENCES for k in t.refs), t.key
        assert all(h in HYPOTHESIS_BY_ID for h in t.hypotheses), t.key
        assert t.tournament_key is None or t.tournament_key in {x.key for x in DIRECTIONAL_THEORIES}


def test_registered_numbers_match_the_current_plans():
    """If the code changes the plan hashes, the registered numbers must be re-run and updated."""
    assert PREREG.digest == TOURNAMENT_PLAN
    assert default_plan(PREREG.digest).digest == STUDY_PLAN
    assert set(TOURNAMENT) == set(KEYS)
    assert set(DIRECTIONAL) <= {x.key for x in DIRECTIONAL_THEORIES}
