"""Tests for the compute-profile ladder."""

import pytest

from cached_hub import Profile as BaseProfile
from cached_hub.profile import ENV_PROFILE


class Profile(BaseProfile):
    FAST_TEST = 0
    SMALL = 1
    LOW_GPU = 2
    HIGH_GPU = 3


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Each test starts with nothing chosen and nothing in the environment."""
    monkeypatch.delenv(ENV_PROFILE, raising=False)
    Profile.reset()
    yield
    Profile.reset()


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------


def test_a_course_can_subclass_the_base():
    assert [p.name for p in Profile.rungs()] == [
        "FAST_TEST",
        "SMALL",
        "LOW_GPU",
        "HIGH_GPU",
    ]


def test_rungs_are_ordered():
    assert Profile.HIGH_GPU > Profile.LOW_GPU > Profile.SMALL > Profile.FAST_TEST


def test_default_is_the_largest_rung():
    """An unset profile means a full-size run, as an unset TESTING_MODE did."""
    assert Profile.current() is Profile.HIGH_GPU


@pytest.mark.parametrize("written", ["low-gpu", "LOW_GPU", "low gpu", " Low-Gpu "])
def test_environment_names_are_forgiving(monkeypatch, written):
    monkeypatch.setenv(ENV_PROFILE, written)
    assert Profile.current() is Profile.LOW_GPU


def test_an_unparseable_environment_warns_and_is_ignored(monkeypatch, caplog):
    monkeypatch.setenv(ENV_PROFILE, "enormous")
    assert Profile.current() is Profile.HIGH_GPU
    assert "enormous" in caplog.text


def test_set_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_PROFILE, "low-gpu")
    Profile.set(Profile.FAST_TEST)
    assert Profile.current() is Profile.FAST_TEST


def test_detection_is_only_a_default(monkeypatch):
    Profile.set_detected(Profile.LOW_GPU)
    assert Profile.current() is Profile.LOW_GPU
    monkeypatch.setenv(ENV_PROFILE, "small")
    assert Profile.current() is Profile.SMALL


def test_parse_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown profile"):
        Profile.parse("enormous")


# ---------------------------------------------------------------------------
# pick()
# ---------------------------------------------------------------------------


def test_pick_returns_the_value_for_the_active_rung():
    Profile.set(Profile.SMALL)
    assert Profile.pick(fast_test="tiny", small="s", low_gpu="l") == "s"


def test_pick_falls_back_downwards():
    """HIGH_GPU is unspecified, so it inherits the largest rung below it."""
    Profile.set(Profile.HIGH_GPU)
    assert Profile.pick(fast_test="tiny", small="s", low_gpu="l") == "l"


def test_pick_falls_back_upwards_when_nothing_is_below():
    Profile.set(Profile.FAST_TEST)
    assert Profile.pick(high_gpu="only") == "only"


def test_a_positional_default_covers_every_unspecified_rung():
    Profile.set(Profile.HIGH_GPU)
    assert Profile.pick(200, fast_test=8, small=40) == 200
    Profile.set(Profile.LOW_GPU)
    assert Profile.pick(200, fast_test=8, small=40) == 200


def test_a_positional_default_does_not_shadow_a_named_rung():
    Profile.set(Profile.FAST_TEST)
    assert Profile.pick(200, fast_test=8, small=40) == 8


def test_pick_resolves_at_call_time_not_at_import():
    """Switching profile and re-running a cell has to actually change things."""
    Profile.set(Profile.HIGH_GPU)
    ladder = dict(fast_test=8, small=40, low_gpu=200)
    assert Profile.pick(**ladder) == 200
    Profile.set(Profile.FAST_TEST)
    assert Profile.pick(**ladder) == 8


def test_pick_rejects_an_unknown_rung():
    with pytest.raises(TypeError, match="unknown rung 'gpu'"):
        Profile.pick(1, gpu=2)


def test_pick_needs_at_least_one_rung():
    with pytest.raises(TypeError, match="at least one rung"):
        Profile.pick(1)


def test_pick_takes_at_most_one_default():
    with pytest.raises(TypeError, match="at most one positional default"):
        Profile.pick(1, 2, small=3)


def test_pick_accepts_falsy_values():
    """`pick(small=0)` must give 0, not fall through to another rung."""
    Profile.set(Profile.SMALL)
    assert Profile.pick(10, small=0) == 0
    assert Profile.pick(10, small=False) is False


# ---------------------------------------------------------------------------
# Hardware-driven detection
# ---------------------------------------------------------------------------


class Detecting(BaseProfile):
    SMALL = 0
    LOW_GPU = 1
    HIGH_GPU = 2

    @classmethod
    def detect(cls, hardware=None):
        if hardware is None or hardware.backend == "cpu":
            return cls.SMALL
        if hardware.backend == "cuda" and hardware.total_memory_gb >= 32:
            return cls.HIGH_GPU
        return cls.LOW_GPU


class FakeHardware:
    def __init__(self, backend, total_memory_gb=0.0):
        self.backend = backend
        self.total_memory_gb = total_memory_gb


@pytest.mark.parametrize(
    "hardware,expected",
    [
        (FakeHardware("cpu"), "SMALL"),
        (FakeHardware("mps", 128.0), "LOW_GPU"),
        (FakeHardware("cuda", 16.0), "LOW_GPU"),
        (FakeHardware("cuda", 80.0), "HIGH_GPU"),
        (None, "SMALL"),
    ],
)
def test_a_course_maps_hardware_to_a_rung(hardware, expected):
    assert Detecting.detect(hardware).name == expected


def test_the_base_detect_ignores_hardware():
    assert Profile.detect(FakeHardware("cuda", 80.0)) is Profile.HIGH_GPU


def test_two_ladders_do_not_see_each_other_s_choice():
    """A stale value from another ladder must not leak into this one."""
    Detecting.set(Detecting.SMALL)
    assert Profile.current() is Profile.HIGH_GPU
