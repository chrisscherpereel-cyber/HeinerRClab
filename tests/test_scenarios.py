"""The scenario module: preset data and the scenario builder, independent of Streamlit.

heiner_abm.scenarios holds what ui/common.py used to carry inline. These tests pin the two properties the move was
meant to buy: the builder is a pure function of a settings mapping, and the app still reaches exactly the same
objects through ui.common.
"""
import importlib
import sys

import pytest

from heiner_abm.params import Scenario
from heiner_abm.scenarios import DEFAULTS, PRESETS, PRESET_INFO, build_scenario


def _settings(preset=None):
    vals = dict(DEFAULTS)
    if preset is not None:
        vals.update(PRESETS[preset])
    return vals


def test_the_module_does_not_pull_in_streamlit():
    """Importing the scenario definitions must not drag the interface in: simulation code imports this module."""
    mod = sys.modules["heiner_abm.scenarios"]
    src = open(mod.__file__, encoding="utf-8").read()
    assert "streamlit" not in src


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_builds_a_scenario(preset):
    scn = build_scenario(_settings(preset))
    assert isinstance(scn, Scenario)
    assert len(scn.firms) >= 1
    assert scn.periods > 0 and 0 <= scn.burn_in < scn.periods


def test_the_builder_is_pure_and_repeatable():
    """Same settings in, equal scenario out, and the caller's mapping is left alone."""
    vals = _settings()
    before = dict(vals)
    first, second = build_scenario(vals), build_scenario(vals)
    assert vals == before, "build_scenario must not mutate the settings it is given"
    assert first == second


def test_every_preset_setting_is_a_known_setting():
    """A preset key that DEFAULTS does not define would be silently ignored by the sidebar."""
    for name, preset in PRESETS.items():
        unknown = set(preset) - set(DEFAULTS)
        assert not unknown, f"preset {name!r} sets unknown keys {sorted(unknown)}"


def test_every_preset_is_described():
    assert set(PRESET_INFO) == set(PRESETS)


def test_ui_common_re_exports_the_same_objects():
    """The sidebar must read these definitions, not keep a second copy of them."""
    common = importlib.import_module("ui.common")
    assert common.DEFAULTS is DEFAULTS
    assert common.PRESETS is PRESETS
    assert common.PRESET_INFO is PRESET_INFO
    assert common.build_scenario is build_scenario
