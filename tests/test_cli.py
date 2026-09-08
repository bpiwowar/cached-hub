"""Tests for the command line interface."""

import os
import sys
import textwrap
from unittest.mock import patch

import pytest

from cached_hub.cli import load_resources, main
from cached_hub.config import ENV_PATH

MODULE = textwrap.dedent(
    """
    from cached_hub import FunctionalResource

    CALLS = []

    def _dl(name):
        def fn():
            CALLS.append(name)
            return f"Downloaded {name}"
        return fn

    SHARED = FunctionalResource("test", "shared", "Shared thing", _dl("shared"))
    RESOURCES = {
        "practical1": [SHARED, FunctionalResource("test", "a", "Thing A", _dl("a"))],
        "practical2": [
            SHARED,
            FunctionalResource("test", "opt", "Optional", _dl("opt"), optional=True),
        ],
    }

    def get_resources():
        return RESOURCES
    """
)


@pytest.fixture
def course(tmp_path, monkeypatch):
    (tmp_path / "fake_course.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_course", None)
    yield "fake_course"
    sys.modules.pop("fake_course", None)


def test_load_resources_attribute_and_callable(course):
    assert set(load_resources(f"{course}:RESOURCES")) == {"practical1", "practical2"}
    assert set(load_resources(f"{course}:get_resources")) == {
        "practical1",
        "practical2",
    }


def test_load_resources_bad_spec():
    with pytest.raises(SystemExit):
        load_resources("no_colon")


def test_info(capsys):
    with patch.dict(os.environ, {ENV_PATH: "/data"}, clear=True):
        assert main(["info"]) == 0
    assert "/data" in capsys.readouterr().out


def test_list(course, capsys):
    assert main(["list", "--from", f"{course}:RESOURCES"]) == 0
    out = capsys.readouterr().out
    assert "practical1:" in out
    assert "- shared: Shared thing" in out
    assert "- opt: Optional (optional)" in out


def test_download_dedup_and_optional(course, tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        assert main(["download", "--from", f"{course}:get_resources"]) == 0
    assert sys.modules[course].CALLS == ["shared", "a"]


def test_download_section_key_optional(course):
    mod_calls = lambda: sys.modules[course].CALLS  # noqa: E731
    with patch.dict(os.environ, {}, clear=True):
        main(["download", "--from", f"{course}:RESOURCES", "--section", "practical2"])
        assert mod_calls() == ["shared"]
        main(["download", "--from", f"{course}:RESOURCES", "--optional"])
        assert mod_calls() == ["shared", "shared", "a", "opt"]
        mod_calls().clear()
        main(["download", "--from", f"{course}:RESOURCES", "--key", "opt"])
        assert mod_calls() == ["opt"]
