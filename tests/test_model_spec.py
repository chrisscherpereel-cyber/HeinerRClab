"""The model specification table covers every agent the laboratory runs."""
from heiner_abm.arena import ALL_DESIGNS
from heiner_abm.model_spec import COLUMNS, all_specs, table
from heiner_abm.params import RULE_TYPES, SELECTION_RULES
from heiner_abm.tasks import LAYERS


def test_every_agent_has_a_complete_row():
    keys = {s.key for s in all_specs()}
    assert set(RULE_TYPES) <= keys
    assert set(SELECTION_RULES) <= keys
    assert set(ALL_DESIGNS) <= keys
    assert set(LAYERS) <= keys
    df = table()
    assert list(df.columns) == list(COLUMNS)
    assert not (df[["Objective", "Information", "Actions", "Feedback"]] == "").any().any()


def test_researcher_variants_are_labeled():
    rows = {s.key: s for s in all_specs()}
    assert "ORACLE" in rows["heiner_m_oracle"].limits and "TRUE" in rows["heiner_p_true"].feedback
    assert "ORACLE" in rows["rc_oracle"].assumptions
    assert "not price-setting bertrand" in rows["Bertrand"].limits.lower()
