"""Compute profiles: how much work a notebook should do on this machine.

A *profile* answers one question — how much compute to spend — and nothing
else. What the machine actually offers (CUDA or MPS, which dtype, whether
``bitsandbytes`` is importable) is a separate concern, handled by whatever
library detects hardware; the two are deliberately kept apart.

The ladder itself is course material, not library material: a practical on
retrieval and one on diffusion do not want the same rungs. So this module
provides only the machinery, and each course declares its own rungs by
subclassing :class:`Profile`::

    from cached_hub import Profile as BaseProfile

    class Profile(BaseProfile):
        FAST_TEST = 0
        SMALL = 1
        LOW_GPU = 2
        HIGH_GPU = 3

Values order the rungs, so ``Profile.current() >= Profile.LOW_GPU`` is
meaningful. Notebooks then size themselves with :meth:`Profile.pick`::

    MODEL_NAME = Profile.pick(
        fast_test="HuggingFaceTB/SmolLM2-135M-Instruct",
        small="Qwen/Qwen2.5-0.5B-Instruct",
        low_gpu="Qwen/Qwen2.5-1.5B-Instruct",
    )
    n_queries = Profile.pick(200, fast_test=8, small=40)

``pick`` resolves when it is called, never at import, so switching profile from
the widget and re-running a cell does what it looks like it does.

Keeping this class here rather than in the course is what lets
:mod:`cached_hub.scan` read a ladder out of a source file without importing it:
it finds the ``class X(Profile)`` statement, learns the rung names from it, and
can then resolve every ``X.pick(...)`` call into the resources it may load.

Environment variables
---------------------
``NOTEBOOK_PROFILE``
    Initial rung, by name (``fast-test``, ``FAST_TEST`` and ``fast test`` are
    all accepted). Unparseable values warn and are ignored. This is a starting
    point, not a lock: :meth:`Profile.set` still wins afterwards.
``NOTEBOOK_PROFILE_WIDGET``
    Set to a falsy value (``0``, ``no``, ``off``, ``false``) to suppress the
    in-notebook chooser.
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("cached_hub")

ENV_PROFILE = "NOTEBOOK_PROFILE"
ENV_PROFILE_WIDGET = "NOTEBOOK_PROFILE_WIDGET"

_FALSY = {"0", "no", "off", "false", "none", ""}

#: Rung chosen explicitly through :meth:`Profile.set`, highest precedence.
_explicit: Optional["Profile"] = None
#: Rung proposed by hardware detection, used when nothing else applies.
_detected: Optional["Profile"] = None
#: Most recently used ladder, so a widget can be shown without being handed one.
_ladder: Optional[type] = None


def _normalise(name: str) -> str:
    """Fold a rung name to its canonical form (``fast-test`` -> ``FAST_TEST``)."""
    return name.strip().upper().replace("-", "_").replace(" ", "_")


def is_notebook() -> bool:
    """Return True when running under an IPython kernel (Jupyter, Colab)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ in (
        "ZMQInteractiveShell",
        "Shell",  # Colab
    )


def _widgets_enabled() -> bool:
    value = os.environ.get(ENV_PROFILE_WIDGET)
    return value is None or value.strip().lower() not in _FALSY


class Profile(IntEnum):
    """Base class for a course's profile ladder — subclass it to add rungs.

    Deliberately memberless: Python allows subclassing an enumeration only as
    long as it declares no members of its own, which is exactly what makes this
    usable as a base.
    """

    @classmethod
    def rungs(cls) -> Sequence["Profile"]:
        """The ladder, lowest rung first."""
        return sorted(cls, key=int)

    @classmethod
    def parse(cls, value: Any) -> "Profile":
        """Resolve a member, a rung name or a rung value to a member."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[_normalise(value)]
            except KeyError:
                raise ValueError(
                    f"unknown profile {value!r}; expected one of "
                    f"{', '.join(p.name for p in cls.rungs())}"
                ) from None
        return cls(value)

    @classmethod
    def default(cls) -> "Profile":
        """Rung to use when nothing else says otherwise: the largest one.

        Matching the historical behaviour of an unset ``TESTING_MODE`` — do the
        full-size thing unless told to cut down.
        """
        return cls.rungs()[-1]

    @classmethod
    def detect(cls, hardware: Any = None) -> "Profile":
        """Rung suggested by the machine. Override to map hardware to a rung.

        ``hardware`` is whatever the caller detects (an object exposing
        ``backend`` and ``total_memory_gb``, say) or ``None``. The base
        implementation ignores it.
        """
        return cls.default()

    @classmethod
    def from_env(cls) -> Optional["Profile"]:
        """Rung named by ``NOTEBOOK_PROFILE``, or ``None`` when unset/invalid."""
        raw = os.environ.get(ENV_PROFILE)
        if raw is None or not raw.strip():
            return None
        try:
            return cls.parse(raw)
        except ValueError as error:
            logger.warning("ignoring %s: %s", ENV_PROFILE, error)
            return None

    @classmethod
    def current(cls) -> "Profile":
        """The active rung: explicit choice, else environment, else detection."""
        global _ladder
        _ladder = cls
        if _explicit is not None and isinstance(_explicit, cls):
            return _explicit
        from_env = cls.from_env()
        if from_env is not None:
            return from_env
        if _detected is not None and isinstance(_detected, cls):
            return _detected
        return cls.default()

    @classmethod
    def set(cls, value: Any) -> "Profile":
        """Choose the active rung explicitly. Wins over the environment."""
        global _explicit, _ladder
        _explicit = cls.parse(value)
        _ladder = cls
        logger.info("profile: %s", _explicit.name)
        return _explicit

    @classmethod
    def set_detected(cls, value: Any) -> "Profile":
        """Record the rung hardware detection suggests, as a weak default."""
        global _detected, _ladder
        _detected = cls.parse(value)
        _ladder = cls
        return _detected

    @classmethod
    def reset(cls) -> None:
        """Forget the explicit and detected rungs (used by tests)."""
        global _explicit, _detected
        _explicit = None
        _detected = None

    @classmethod
    def pick(cls, *default: Any, **by_rung: Any) -> Any:
        """Choose a value for the active rung.

        ``by_rung`` maps rung names, lowercased, to values. Resolution:

        1. the value given for the active rung, if any;
        2. otherwise the positional ``default``, if one was given — it stands
           for every rung left unspecified;
        3. otherwise the nearest specified rung *below* the active one, and
           failing that the nearest above.

        Rule 3 is what makes a partial ladder read well: a call giving only
        ``fast_test``, ``small`` and ``low_gpu`` leaves the largest rung with
        the ``low_gpu`` value rather than an error.
        """
        if len(default) > 1:
            raise TypeError(
                f"pick() takes at most one positional default, got {len(default)}"
            )
        if not by_rung:
            raise TypeError("pick() needs at least one rung, e.g. pick(fast_test=8)")

        values: Dict[int, Any] = {}
        for name, value in by_rung.items():
            try:
                rung = cls[_normalise(name)]
            except KeyError:
                raise TypeError(
                    f"pick() got an unknown rung {name!r}; expected one of "
                    f"{', '.join(p.name.lower() for p in cls.rungs())}"
                ) from None
            values[int(rung)] = value

        active = int(cls.current())
        if active in values:
            return values[active]
        if default:
            return default[0]

        below: List[int] = [v for v in values if v < active]
        if below:
            return values[max(below)]
        return values[min(values)]

    @classmethod
    def select(cls, *, auto: bool = False) -> "Profile":
        """Show the in-notebook chooser. No-op outside a notebook.

        With ``auto``, stay silent when ``NOTEBOOK_PROFILE`` was set: the
        environment already made the choice, and a widget contradicting it is
        worse than no widget.
        """
        active = cls.current()
        if not is_notebook() or not _widgets_enabled():
            return active
        if auto and cls.from_env() is not None:
            return active
        try:
            import ipywidgets
            from IPython.display import display
        except ImportError:
            logger.debug("ipywidgets is not installed, no profile chooser")
            return active

        buttons = ipywidgets.ToggleButtons(
            options=[(p.name, p) for p in cls.rungs()],
            value=active,
            description="Profile:",
        )
        label = ipywidgets.HTML()

        def describe(rung: "Profile") -> str:
            text = (cls.describe(rung) or "").strip()
            return f"<i>{text}</i>" if text else ""

        label.value = describe(active)

        def on_change(change: Any) -> None:
            if change["name"] == "value":
                cls.set(change["new"])
                label.value = describe(change["new"])

        buttons.observe(on_change, names="value")
        display(ipywidgets.VBox([buttons, label]))
        return active

    @classmethod
    def describe(cls, rung: "Profile") -> str:
        """One-line description of a rung, shown next to the chooser."""
        return ""


def current_ladder() -> Optional[type]:
    """The most recently used :class:`Profile` subclass, if any."""
    return _ladder
