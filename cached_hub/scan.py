"""Read resource declarations out of the sources that load them.

``--from`` imports a hand-written mapping, which is free to drift away from the
code it describes. This module closes that gap by parsing both sides and
comparing them:

- **the sources** — every ``load_hf_model`` / ``load_hf_tokenizer`` /
  ``load_hf_processor`` / ``load_hf_dataset`` / ``HFModel`` /
  ``pt.get_dataset`` call, with module-level string constants resolved
  (``MODEL_NAME = "gpt2"`` … ``load_hf_model(MODEL_NAME, …)``). Plain
  ``load_dataset`` / ``Class.from_pretrained`` calls are collected too, but as
  *bypasses*: they do not go through the cache, so they are reported and never
  compared;
- **the declaration** — the ``{section: [resource, …]}`` mapping of a Python
  file, parsed rather than imported, so that class names and ``optional`` flags
  are read exactly as written.

Sections are the source file names: ``sources/02-generation.py`` describes the
``02-generation`` section, which is what ``--section`` takes.

A constant rebound under a guard (``if test_mode:`` by default, see
``guards``) yields a second, *optional* resource: that is the small stand-in a
course swaps in when testing, and it has no business being downloaded for a
classroom.

A :class:`~cached_hub.profile.Profile` ladder says the same thing in one
expression, and is read the same way::

    class Profile(BaseProfile):
        FAST_TEST = 0
        SMALL = 1
        LOW_GPU = 2

    MODEL = Profile.pick(fast_test="…/SmolLM2-135M", low_gpu="Qwen/…-1.5B")

The ``class X(Profile)`` statement is what gives the rung names and their
order, so a ladder is understood without importing anything. Of the models a
``pick`` may return, the largest rung's is required — it is what the ladder
resolves to when nothing selects a profile — and the smaller ones are
optional. A positional default plays the largest rung's part, so when one is
given every keyword rung is optional.

What a scan cannot know is left to the human: descriptions, and the dataset
*splits* — code that loads every split says nothing about their names. Splits
are reported, never compared.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ScannedResource",
    "compare",
    "emit_section",
    "parse_declaration",
    "scan_paths",
]

#: Default names whose ``if <name>:`` block marks an instructor-only stand-in.
DEFAULT_GUARDS = ("test_mode", "TESTING_MODE")

#: ``cached_hub`` loader -> (kind, class keyword, default class).
LOADERS = {
    "load_hf_model": ("model", "model_class", "AutoModel"),
    "load_hf_tokenizer": ("tokenizer", "tokenizer_class", "AutoTokenizer"),
    "load_hf_processor": ("processor", "processor_class", "AutoProcessor"),
}

#: Declaration helper -> (kind, positional parameter names).
MAKERS = {
    "make_hf_model_resource": ("model", ("key", "description", "model_class")),
    "make_hf_tokenizer_resource": (
        "tokenizer",
        ("key", "description", "tokenizer_class"),
    ),
    "make_hf_processor_resource": (
        "processor",
        ("key", "description", "processor_class"),
    ),
    "make_hf_dataset_resource": ("dataset", ("key", "splits", "description", "name")),
    "make_pyterrier_dataset_resource": ("pyterrier", ("key", "description")),
    "make_datamaestro_resource": ("datamaestro", ("key", "description")),
}

#: Kind -> declaration helper that builds it.
MAKER_OF_KIND = {
    "model": "make_hf_model_resource",
    "tokenizer": "make_hf_tokenizer_resource",
    "processor": "make_hf_processor_resource",
    "dataset": "make_hf_dataset_resource",
    "pyterrier": "make_pyterrier_dataset_resource",
    "datamaestro": "make_datamaestro_resource",
}

#: Keyword naming the class, per kind (kinds without one are absent).
CLASS_KEYWORD = {
    "model": "model_class",
    "tokenizer": "tokenizer_class",
    "processor": "processor_class",
}


@dataclass(frozen=True)
class ScannedResource:
    """A resource a source file loads, or a declaration lists."""

    kind: str  # model, tokenizer, processor, dataset, pyterrier, datamaestro
    key: str  # model id, ``dataset[config]``, ``irds:…``
    cls: Optional[str] = None  # transformers class name, when relevant
    optional: bool = False  # loaded only under a guard / declared optional
    splits: Optional[Tuple[str, ...]] = None  # datasets only, never compared
    bypass: bool = False  # loaded outside cached_hub, hence uncacheable

    @property
    def identity(self) -> Tuple[str, str]:
        """What ``cached_hub`` considers the same resource."""
        return (self.kind, self.key)

    def __str__(self) -> str:
        out = f"{self.kind} {self.key}"
        if self.cls:
            out += f" ({self.cls})"
        if self.splits:
            out += f" splits={'+'.join(self.splits)}"
        if self.optional:
            out += " [optional]"
        return out


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@dataclass
class _Constants:
    """String constants of one module, with their guarded variants.

    ``values[name]`` is ``[(value, optional), …]`` in source order: the first
    entry is the value a reader sees, any later one comes from a rebinding
    under a guard.
    """

    guards: Tuple[str, ...] = DEFAULT_GUARDS
    values: Dict[str, List[Tuple[object, bool]]] = field(default_factory=dict)
    #: Ladder class name -> ``{RUNG_NAME: rank}``, from ``class X(Profile)``.
    ladders: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, name: str, value: object, optional: bool) -> None:
        entries = self.values.setdefault(name, [])
        if all(existing != value for existing, _ in entries):
            entries.append((value, optional))

    def pick_values(self, node: ast.expr) -> List[Tuple[object, bool]]:
        """Every value a ``<Ladder>.pick(...)`` call may yield.

        The largest rung mentioned is what the ladder resolves to when nothing
        selects a profile, so it is the one a full run needs and comes out
        required; the smaller rungs are stand-ins and come out optional. A
        positional default stands for every unspecified rung, the largest
        included, so it is required too — and then no keyword rung is.
        """
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return []
        if node.func.attr != "pick":
            return []
        rungs = self.ladders.get(_func_name(node.func.value))
        if not rungs:
            return []

        ranked: List[Tuple[int, object]] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            rank = rungs.get(_normalise_rung(keyword.arg))
            value = _literal(keyword.value)
            if rank is not None and value is not None:
                ranked.append((rank, value))
        if not ranked:
            return []

        out: List[Tuple[object, bool]] = []
        for default in node.args[:1]:
            value = _literal(default)
            if value is not None:
                out.append((value, False))
        largest = max(rank for rank, _ in ranked)
        for rank, value in sorted(ranked, key=lambda item: -item[0]):
            out.append((value, bool(out) or rank < largest))
        return out

    def resolve(self, node: ast.expr) -> List[Tuple[object, bool]]:
        """Every value ``node`` may take, each with its ``optional`` flag."""
        if isinstance(node, ast.Constant):
            return [(node.value, False)]
        if isinstance(node, ast.Name):
            return list(self.values.get(node.id, []))
        if isinstance(node, ast.Call):
            return self.pick_values(node)
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [self.resolve(elt) for elt in node.elts]
            if not items or any(not item for item in items):
                return []
            return [([item[0][0] for item in items], False)]
        return []

    def first(self, node: Optional[ast.expr]) -> Optional[object]:
        """The value a reader would assume, or ``None`` if not a constant."""
        if node is None:
            return None
        candidates = self.resolve(node)
        return candidates[0][0] if candidates else None


def _literal(node: ast.expr) -> Optional[object]:
    """A string, or a list of strings, or ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if items and len(items) == len(node.elts):
            return items
    return None


def _normalise_rung(name: str) -> str:
    """``fast_test`` / ``fast-test`` -> ``FAST_TEST``, as :mod:`cached_hub.profile`."""
    return name.strip().upper().replace("-", "_").replace(" ", "_")


def _profile_base_aliases(tree: ast.AST) -> set:
    """Names under which :class:`cached_hub.Profile` was imported.

    ``from cached_hub import Profile as BaseProfile`` yields ``{"BaseProfile"}``,
    so the ``class Profile(BaseProfile)`` below it is recognised as a ladder
    even though it shadows the imported name.
    """
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in ("cached_hub", "cached_hub.profile"):
                for alias in node.names:
                    if alias.name == "Profile":
                        aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cached_hub" and alias.asname is None:
                    aliases.add("cached_hub.Profile")
    return aliases


def _collect_ladders(tree: ast.AST) -> Dict[str, Dict[str, int]]:
    """``{ladder class name: {RUNG: rank}}`` for every ``class X(Profile)``.

    Read straight off the class statement, so a ladder is understood without
    importing the module that declares it — the reason :class:`Profile` lives
    in this package rather than in each course.
    """
    aliases = _profile_base_aliases(tree)
    ladders: Dict[str, Dict[str, int]] = {}
    if not aliases:
        return ladders

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.add(base.id)
            elif isinstance(base, ast.Attribute):
                bases.add(f"{_func_name(base.value)}.{base.attr}")
        if not bases & aliases:
            continue
        rungs: Dict[str, int] = {}
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            value = statement.value
            if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
                if isinstance(value.value, int):
                    rungs[_normalise_rung(target.id)] = value.value
        if rungs:
            ladders[node.name] = rungs
    return ladders


def _collect_constants(
    tree: ast.AST,
    guards: Sequence[str],
    *,
    ladders: Optional[Dict[str, Dict[str, int]]] = None,
) -> _Constants:
    constants = _Constants(tuple(guards))
    # Ladders have to be in place *before* the walk: resolving `X.pick(...)` on
    # the right-hand side of an assignment happens during it.
    constants.ladders = dict(ladders or {})
    constants.ladders.update(_collect_ladders(tree))

    def guarded(test: ast.expr) -> bool:
        if isinstance(test, ast.Name):
            return test.id in constants.guards
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
            return test.left.id in constants.guards
        return False

    def walk(node: ast.AST, under_guard: bool) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _literal(node.value)
                if value is not None:
                    constants.add(target.id, value, under_guard)
                else:
                    # `MODEL = Profile.pick(low_gpu="…", small="…")`: every rung
                    # is a model this notebook may load, the smaller ones only
                    # when someone asks for a smaller profile.
                    for picked, optional in constants.pick_values(node.value):
                        constants.add(target.id, picked, under_guard or optional)
        if isinstance(node, ast.If):
            # `if test_mode:` swaps in a stand-in; `else` stays on the main path
            for stmt in node.body:
                walk(stmt, under_guard or guarded(node.test))
            for stmt in node.orelse:
                walk(stmt, under_guard)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, under_guard)

    walk(tree, False)
    return constants


def _class_name(node: Optional[ast.expr]) -> Optional[str]:
    """``AutoModel`` / ``transformers.GPT2Tokenizer`` -> the class name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value  # declarations pass the class name as a string
    return None


def _keyword(call: ast.Call, name: str) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _func_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _kind_of_class(class_name: str) -> str:
    """Guess what a ``Class.from_pretrained`` call loads, from the class name."""
    if "Tokenizer" in class_name:
        return "tokenizer"
    if "Processor" in class_name or "FeatureExtractor" in class_name:
        return "processor"
    return "model"


# ---------------------------------------------------------------------------
# Scanning sources
# ---------------------------------------------------------------------------


class _Scanner(ast.NodeVisitor):
    def __init__(self, constants: _Constants) -> None:
        self.constants = constants
        self.resources: List[ScannedResource] = []
        self.imports: List[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _func_name(node.func)
        if name in LOADERS:
            self._pretrained(node, *LOADERS[name])
        elif name == "HFModel":
            self._hf_model(node)
        elif name == "load_hf_dataset":
            self._dataset(node)
        elif name == "load_dataset":
            self._dataset(node, bypass=True)
        elif name == "get_dataset":
            self._pyterrier(node)
        elif name == "from_pretrained":
            self._from_pretrained(node)
        elif name == "prepare_dataset":
            self._datamaestro(node)
        self.generic_visit(node)

    def _add(self, resource: ScannedResource) -> None:
        self.resources.append(resource)

    def _pretrained(self, node: ast.Call, kind: str, kw: str, default: str) -> None:
        if not node.args:
            return
        class_node = node.args[1] if len(node.args) > 1 else _keyword(node, kw)
        cls = _class_name(class_node) or default
        for value, optional in self.constants.resolve(node.args[0]):
            if isinstance(value, str):
                self._add(ScannedResource(kind, value, cls, optional))

    def _hf_model(self, node: ast.Call) -> None:
        """``HFModel(id, tokenizer_cls=…, model_cls=…)`` loads both."""
        if not node.args:
            return
        model_node = node.args[2] if len(node.args) > 2 else _keyword(node, "model_cls")
        tok_node = (
            node.args[1] if len(node.args) > 1 else _keyword(node, "tokenizer_cls")
        )
        model_cls = _class_name(model_node) or "AutoModel"
        tok_cls = _class_name(tok_node) or "AutoTokenizer"
        for value, optional in self.constants.resolve(node.args[0]):
            if not isinstance(value, str):
                continue
            self._add(ScannedResource("model", value, model_cls, optional))
            self._add(ScannedResource("tokenizer", value, tok_cls, optional))

    def _from_pretrained(self, node: ast.Call) -> None:
        """``AutoModelForSequenceClassification.from_pretrained("gpt2")``."""
        if not node.args or not isinstance(node.func, ast.Attribute):
            return
        cls = _class_name(node.func.value)
        if cls is None or not cls[:1].isupper():
            return  # `model.from_pretrained` / an instance: not a class
        for value, optional in self.constants.resolve(node.args[0]):
            if isinstance(value, str):
                self._add(
                    ScannedResource(
                        _kind_of_class(cls), value, cls, optional, bypass=True
                    )
                )

    def _dataset(self, node: ast.Call, bypass: bool = False) -> None:
        """``load_hf_dataset(id, config, split=…)`` — 2nd arg is the config."""
        if not node.args:
            return
        config_node = node.args[1] if len(node.args) > 1 else _keyword(node, "name")
        config = self.constants.first(config_node)
        split = self.constants.first(
            _keyword(node, "split") or _keyword(node, "splits")
        )
        if split is None:
            splits = None
        elif isinstance(split, str):
            splits = (split,)
        else:
            splits = tuple(split)
        for value, optional in self.constants.resolve(node.args[0]):
            if not isinstance(value, str):
                continue
            key = f"{value}[{config}]" if isinstance(config, str) else value
            self._add(
                ScannedResource(
                    "dataset", key, None, optional, splits=splits, bypass=bypass
                )
            )

    def _datamaestro(self, node: ast.Call) -> None:
        if not node.args:
            return
        value = self.constants.first(node.args[0])
        if isinstance(value, str):
            self._add(ScannedResource("datamaestro", value))

    def _pyterrier(self, node: ast.Call) -> None:
        if not node.args:
            return
        value = self.constants.first(node.args[0])
        if isinstance(value, str) and value.startswith("irds:"):
            self._add(ScannedResource("pyterrier", value))


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def scan_file(
    path: Path,
    guards: Sequence[str] = DEFAULT_GUARDS,
    *,
    ladders: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[List[ScannedResource], List[str]]:
    """Resources loaded by one file, and the modules it imports.

    ``ladders`` adds profile ladders declared elsewhere — a course usually
    declares one in a helper module and uses it in the notebooks that import it,
    so the rung names have to come in from outside the file being scanned.
    """
    tree = _parse(path)
    constants = _collect_constants(tree, guards, ladders=ladders)
    scanner = _Scanner(constants)
    scanner.visit(tree)
    return scanner.resources, scanner.imports


def _module_file(module: str, search_paths: Sequence[Path]) -> Optional[Path]:
    relative = Path(*module.split("."))
    for root in search_paths:
        for candidate in (
            root / relative.with_suffix(".py"),
            root / relative / "__init__.py",
        ):
            if candidate.is_file():
                return candidate
    return None


def _related_files(source: Path, search_paths: Sequence[Path]) -> List[Path]:
    """``source`` plus every module it imports, transitively, under the roots."""
    files = [source]
    pending = _module_imports(_parse(source))
    seen = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        helper = _module_file(module, search_paths)
        if helper is None or helper in files:
            continue
        files.append(helper)
        pending.extend(_module_imports(_parse(helper)))
    return files


def _module_imports(tree: ast.AST) -> List[str]:
    return [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    ]


def scan_paths(
    paths: Sequence[Path],
    *,
    guards: Sequence[str] = DEFAULT_GUARDS,
    search_paths: Sequence[Path] = (),
) -> Dict[str, List[ScannedResource]]:
    """Scan files (or every ``*.py`` of a directory), keyed by file name.

    ``search_paths`` are roots under which imported modules are looked up and
    scanned as part of the importing section — for course code split between a
    notebook and a helper module.
    """
    sources: List[Path] = []
    for path in paths:
        if path.is_dir():
            sources.extend(
                p for p in sorted(path.glob("*.py")) if not p.name.startswith("_")
            )
        else:
            sources.append(path)

    scanned: Dict[str, List[ScannedResource]] = {}
    for source in sources:
        related = _related_files(source, search_paths)
        # Pool the ladders over the whole import graph first: the `class
        # X(Profile)` statement is typically in a helper module while the
        # `X.pick(...)` calls are in the notebook that imports it, and neither
        # file can be understood alone.
        ladders: Dict[str, Dict[str, int]] = {}
        for path in related:
            ladders.update(_collect_ladders(_parse(path)))

        resources: List[ScannedResource] = []
        for path in related:
            more, _ = scan_file(path, guards, ladders=ladders)
            resources.extend(more)
        scanned[source.stem] = _dedup(resources)
    return scanned


def _dedup(resources: Iterable[ScannedResource]) -> List[ScannedResource]:
    """One entry per identity; the strongest claim wins."""
    by_identity: Dict[Tuple[str, str], ScannedResource] = {}
    for resource in resources:
        current = by_identity.get(resource.identity)
        if current is None:
            by_identity[resource.identity] = resource
            continue
        # A required load beats a guarded one, and a cached load beats a bypass
        better = (not resource.optional and current.optional) or (
            not resource.bypass and current.bypass
        )
        if better:
            by_identity[resource.identity] = resource
        elif current.splits is None and resource.splits is not None:
            by_identity[resource.identity] = resource
    return list(by_identity.values())


# ---------------------------------------------------------------------------
# Parsing a declaration
# ---------------------------------------------------------------------------


def parse_declaration(
    path: Path, attribute: str = "RESOURCES"
) -> Dict[str, List[ScannedResource]]:
    """Read a ``{section: [resource, …]}`` mapping from a file, without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _collect_constants(tree, ())

    mapping: Optional[ast.Dict] = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == attribute
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                mapping = node.value
    if mapping is None:
        raise ValueError(f"{path}: no `{attribute} = {{…}}` mapping found")

    declared: Dict[str, List[ScannedResource]] = {}
    for key_node, value_node in zip(mapping.keys, mapping.values):
        section = key_node.value if isinstance(key_node, ast.Constant) else None
        if not isinstance(section, str) or not isinstance(value_node, ast.List):
            continue
        resources = []
        for call in value_node.elts:
            if isinstance(call, ast.Call):
                resource = _declared_resource(call, constants)
                if resource is not None:
                    resources.append(resource)
        declared[section] = resources
    return declared


def _declared_resource(
    call: ast.Call, constants: _Constants
) -> Optional[ScannedResource]:
    entry = MAKERS.get(_func_name(call.func))
    if entry is None:
        return None
    kind, positional = entry

    def argument(name: str) -> Optional[ast.expr]:
        if name in positional:
            index = positional.index(name)
            if len(call.args) > index:
                return call.args[index]
        return _keyword(call, name)

    key = constants.first(argument("key"))
    if not isinstance(key, str):
        return None

    class_keyword = CLASS_KEYWORD.get(kind)
    cls = _class_name(argument(class_keyword)) if class_keyword else None

    splits = None
    if kind == "dataset":
        config = constants.first(argument("name"))
        if isinstance(config, str):
            key = f"{key}[{config}]"
        value = constants.first(argument("splits"))
        if isinstance(value, str):
            splits = (value,)
        elif isinstance(value, list):
            splits = tuple(value)

    optional_node = _keyword(call, "optional")
    optional = isinstance(optional_node, ast.Constant) and optional_node.value is True
    return ScannedResource(kind, key, cls, optional, splits=splits)


# ---------------------------------------------------------------------------
# Comparing and emitting
# ---------------------------------------------------------------------------


def compare(
    scanned: Dict[str, List[ScannedResource]],
    declared: Dict[str, List[ScannedResource]],
) -> Tuple[List[str], List[str]]:
    """Compare a scan with a declaration; return ``(errors, warnings)``.

    Bypasses are ignored: a resource loaded outside ``cached_hub`` cannot be
    served from the cache, so declaring it would be a lie.
    """
    errors: List[str] = []
    warnings: List[str] = []

    cached = {
        section: [r for r in resources if not r.bypass]
        for section, resources in scanned.items()
    }

    for section in sorted(set(cached) - set(declared)):
        if cached[section]:
            errors.append(f"{section}: section missing from the declaration")
    for section in sorted(set(declared) - set(cached)):
        errors.append(f"{section}: declared but no such source file")

    for section in sorted(set(cached) & set(declared)):
        loaded = {r.identity: r for r in cached[section]}
        listed = {r.identity: r for r in declared[section]}

        for identity in sorted(loaded.keys() - listed.keys()):
            errors.append(f"{section}: loaded but not declared — {loaded[identity]}")
        for identity in sorted(listed.keys() - loaded.keys()):
            errors.append(f"{section}: declared but never loaded — {listed[identity]}")

        for identity in sorted(loaded.keys() & listed.keys()):
            here, there = loaded[identity], listed[identity]
            if here.cls and there.cls and here.cls != there.cls:
                warnings.append(
                    f"{section}: {here.key} declared as {there.cls}, "
                    f"loaded as {here.cls}"
                )
            if here.optional != there.optional:
                declared_as = "optional" if there.optional else "required"
                loaded_as = "only under a guard" if here.optional else "unconditionally"
                warnings.append(
                    f"{section}: {here.key} declared {declared_as} "
                    f"but loaded {loaded_as}"
                )
    return errors, warnings


def emit_section(section: str, resources: Sequence[ScannedResource]) -> str:
    """A declaration skeleton for one section.

    Descriptions come out as ``"<describe this resource>"`` placeholders and
    dataset splits as a guess: neither can be read off the loading code.
    """
    lines = [f'    "{section}": [']
    for resource in sorted(resources, key=lambda r: (r.kind, r.key)):
        if resource.bypass:
            continue
        key, _, config = resource.key.partition("[")
        config = config.rstrip("]")
        lines.append(f"        {MAKER_OF_KIND[resource.kind]}(")
        lines.append(f'            "{key}",')
        if resource.kind == "dataset":
            splits = list(resource.splits) if resource.splits else ["train"]
            lines.append(f"            {splits!r},  # check the splits")
        lines.append('            "<describe this resource>",')
        if resource.kind == "dataset" and config:
            lines.append(f'            name="{config}",')
        class_keyword = CLASS_KEYWORD.get(resource.kind)
        if class_keyword and resource.cls:
            lines.append(f'            {class_keyword}="{resource.cls}",')
        if resource.optional:
            lines.append("            optional=True,")
        lines.append("        ),")
    lines.append("    ],")
    return "\n".join(lines)
