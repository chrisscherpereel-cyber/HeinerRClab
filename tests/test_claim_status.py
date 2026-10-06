"""The status table separates established theory, reduced-form implementations and proposed extensions."""
from heiner_abm.claim_status import COMPONENTS, STATUS_LABELS, by_status
from heiner_abm.literature import REFERENCES


def test_every_component_has_a_known_status_and_resolvable_sources():
    assert COMPONENTS
    for c in COMPONENTS:
        assert c.status in STATUS_LABELS, c.name
        assert c.name and len(c.what) > 30, c.name
        assert all(k in REFERENCES for k in c.sources), (c.name, c.sources)


def test_every_status_is_used_and_established_claims_cite_their_source():
    for status in STATUS_LABELS:
        assert by_status(status), status
    assert all(c.sources for c in by_status("established"))


def test_component_names_are_unique():
    names = [c.name for c in COMPONENTS]
    assert len(names) == len(set(names))
