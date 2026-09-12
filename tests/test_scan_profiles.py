"""Tests for reading a profile ladder out of the sources that use it."""

import textwrap

import pytest

from cached_hub.cli import main
from cached_hub.scan import scan_paths

LADDER = '''
"""A notebook that sizes itself with a profile ladder."""
from cached_hub import Profile as BaseProfile
from cached_hub import load_hf_dataset, load_hf_model, load_hf_tokenizer


class Profile(BaseProfile):
    FAST_TEST = 0
    SMALL = 1
    LOW_GPU = 2
    HIGH_GPU = 3


MODEL = Profile.pick(
    fast_test="tiny/model",
    small="small/model",
    low_gpu="big/model",
)
JUDGE = Profile.pick("judge/model", fast_test="tiny/judge")

model = load_hf_model(MODEL, AutoModelForCausalLM)
tokenizer = load_hf_tokenizer(MODEL)
judge = load_hf_model(JUDGE, AutoModelForImageTextToText)
inline = load_hf_model(Profile.pick(small="inline/model"))
data = load_hf_dataset("some/corpus", split="train")
n_queries = Profile.pick(200, fast_test=8)
'''

GUARDED = '''
"""A notebook still on the old test_mode guard."""
from cached_hub import load_hf_model

MODEL = "big/model"
if test_mode:
    MODEL = "small/model"

model = load_hf_model(MODEL, AutoModelForCausalLM)
'''


def _write(tmp_path, source):
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "practical.py").write_text(textwrap.dedent(source))
    return tmp_path


@pytest.fixture
def ladder_course(tmp_path):
    return _write(tmp_path, LADDER)


def _scanned(course):
    resources = scan_paths([course / "sources"])["practical"]
    return {(r.kind, r.key): r for r in resources}


def test_a_ladder_yields_every_rung_it_may_load(ladder_course):
    found = _scanned(ladder_course)
    assert ("model", "big/model") in found
    assert ("model", "small/model") in found
    assert ("model", "tiny/model") in found


def test_the_largest_rung_is_required_and_the_stand_ins_optional(ladder_course):
    found = _scanned(ladder_course)
    assert found[("model", "big/model")].optional is False
    assert found[("model", "small/model")].optional is True
    assert found[("model", "tiny/model")].optional is True


def test_a_positional_default_is_the_required_one(ladder_course):
    found = _scanned(ladder_course)
    assert found[("model", "judge/model")].optional is False
    assert found[("model", "tiny/judge")].optional is True


def test_a_ladder_carries_through_to_the_tokenizer(ladder_course):
    found = _scanned(ladder_course)
    assert found[("tokenizer", "big/model")].optional is False
    assert found[("tokenizer", "tiny/model")].optional is True


def test_a_pick_inline_in_a_loader_call_is_resolved(ladder_course):
    found = _scanned(ladder_course)
    assert found[("model", "inline/model")].cls == "AutoModel"


def test_non_string_rungs_are_not_resources(ladder_course):
    """`Profile.pick(200, fast_test=8)` sizes a loop, it loads nothing."""
    found = _scanned(ladder_course)
    assert not [key for key in found if key[1] in {"200", "8"}]


def test_a_ladder_is_read_without_importing_the_module(tmp_path):
    """The rung names come off the class statement, never from an import."""
    course = _write(
        tmp_path, LADDER.replace("import load_hf_dataset", "import nonexistent")
    )
    assert ("model", "big/model") in _scanned(course)


def test_a_pick_on_an_unrelated_class_is_ignored(tmp_path):
    """Without a `class X(Profile)` statement there is no ladder to read."""
    course = _write(
        tmp_path, LADDER.replace("class Profile(BaseProfile):", "class Profile:")
    )
    assert ("model", "big/model") not in _scanned(course)


def test_ladder_emit_is_valid_python(ladder_course, capsys):
    assert main(["scan", str(ladder_course / "sources"), "--emit", "practical"]) == 0
    compile(capsys.readouterr().out, "<emit>", "exec")


def test_the_test_mode_guard_still_works(tmp_path):
    """llm@mind has not migrated yet, so `if test_mode:` must keep scanning."""
    found = _scanned(_write(tmp_path, GUARDED))
    assert found[("model", "big/model")].optional is False
    assert found[("model", "small/model")].optional is True


# --------------------------------------------------------------------------- #
# A ladder declared in one file, used in another
# --------------------------------------------------------------------------- #

SPLIT_LADDER = """
from cached_hub import Profile as BaseProfile


class Profile(BaseProfile):
    FAST_TEST = 0
    SMALL = 1
    LOW_GPU = 2
"""

SPLIT_NOTEBOOK = """
from cached_hub import load_hf_model
from mycourse.profiles import Profile

MODEL = Profile.pick(fast_test="tiny/model", low_gpu="big/model")
model = load_hf_model(MODEL, AutoModelForCausalLM)
"""


def test_a_ladder_declared_in_a_helper_reaches_the_notebook(tmp_path):
    """The usual shape: rungs in a library module, `pick` calls in the notebook.

    Neither file can be understood on its own, so the ladders have to be pooled
    over the import graph before anything is scanned.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "practical.py").write_text(textwrap.dedent(SPLIT_NOTEBOOK))
    helper = tmp_path / "src" / "mycourse"
    helper.mkdir(parents=True)
    (helper / "__init__.py").write_text("")
    (helper / "profiles.py").write_text(textwrap.dedent(SPLIT_LADDER))

    found = {
        (r.kind, r.key): r
        for r in scan_paths([sources], search_paths=[tmp_path / "src"])["practical"]
    }
    assert found[("model", "big/model")].optional is False
    assert found[("model", "tiny/model")].optional is True


def test_without_the_search_path_the_ladder_is_invisible(tmp_path):
    """No search path, no helper module, so no rung names and no resources."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "practical.py").write_text(textwrap.dedent(SPLIT_NOTEBOOK))
    found = scan_paths([sources])["practical"]
    assert not [r for r in found if r.key == "big/model"]
