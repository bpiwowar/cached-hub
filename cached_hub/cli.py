"""Command line: inspect the cache, list, download and audit declared resources.

Resources are given with ``--from module:attribute`` (or ``--from path.py``,
which reads ``RESOURCES`` from a file that need not be importable): the module
is imported and the attribute read from it. It must be a resource mapping
(``{section: [Resource, ...]}``) or a zero-argument callable returning one::

    cached-hub info
    cached-hub list --from llm_course.resources:RESOURCES
    cached-hub list --from src/llm_course/resources.py
    cached-hub download --from llm_course.resources:get_resources --section practical2
    cached-hub download --from llm_course.resources:RESOURCES --key gpt2

``scan`` and ``check`` work the other way round, on the *sources* that load the
resources: they read the loader calls out of the code, so that a declaration
can be written from them (``scan --emit``) and kept honest (``check``)::

    cached-hub scan  sources/
    cached-hub scan  sources/ --emit 02-generation
    cached-hub check sources/ --declaration mycourse/resources.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import config
from .resources import Resources, download_resources, format_resources
from .scan import DEFAULT_GUARDS, compare, emit_section, parse_declaration, scan_paths


def _import_path(path: Path) -> object:
    """Import a ``.py`` file directly, without it being on ``sys.path``."""
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    name = f"_cached_hub_resources_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a bad declaration file, not our problem to fix
        del sys.modules[name]
        raise SystemExit(f"{path}: {exc}") from exc
    return module


def _split_spec(spec: str) -> Tuple[str, str]:
    """``module:attr``, ``file.py`` or ``file.py:attr`` -> (target, attribute)."""
    head, sep, tail = spec.rpartition(":")
    if sep and head.endswith(".py"):
        return head, tail  # file.py:attr
    if spec.endswith(".py"):
        return spec, "RESOURCES"  # file.py, attribute implied
    module_name, sep, attr_path = spec.partition(":")
    if not sep or not attr_path:
        raise SystemExit(
            f"invalid resource spec {spec!r}: expected module:attribute, "
            "path/to/resources.py, or path/to/resources.py:attribute"
        )
    return module_name, attr_path


def load_resources(spec: str) -> Resources:
    """Resolve ``module:attribute`` (or a ``.py`` path) to a resource mapping."""
    target, attr_path = _split_spec(spec)
    if target.endswith(".py"):
        obj: object = _import_path(Path(target))
    else:
        obj = importlib.import_module(target)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if callable(obj):
        obj = obj()
    if not isinstance(obj, dict):
        raise SystemExit(f"{spec} did not yield a {{section: [resources]}} mapping")
    return obj


def _load_all(specs: Sequence[str]) -> List[Tuple[str, Resources]]:
    return [(spec, load_resources(spec)) for spec in specs]


def cmd_info(args: argparse.Namespace) -> int:
    print(config.describe())
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for spec, resources in _load_all(args.resources):
        print(f"{spec}:")
        if not resources:
            print("  (no resources)")
            continue
        print("\n".join(format_resources(resources, indent="  ")))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    root = config.get_cache_path()
    if root is None:
        logging.warning(
            "%s is not set: resources will go to the default HuggingFace cache "
            "(~/.cache/huggingface). Set it to populate a shared cache.",
            config.ENV_PATH,
        )
    else:
        print(f"Cache location: {root}")

    failed: List[Tuple[str, object, Exception]] = []
    with config.hub_online():
        for spec, resources in _load_all(args.resources):
            if args.section and args.section not in resources:
                logging.warning("%s: no section %r", spec, args.section)
                continue
            try:
                download_resources(
                    resources,
                    section=args.section,
                    key=args.key,
                    include_optional=args.optional,
                    keep_going=args.keep_going,
                    failures=failed,
                )
            except Exception:
                if not args.keep_going:
                    raise
                failed.append((spec, None, sys.exc_info()[1]))
                logging.exception("%s: download failed", spec)

    if failed:
        print(f"\n{len(failed)} resource(s) could not be downloaded:")
        for section_name, resource, exc in failed:
            name = getattr(resource, "key", section_name)
            print(f"  - {section_name}/{name}: {exc}")
        return 1
    return 0


def _scan(args: argparse.Namespace):
    search_paths = list(args.search_path or [])
    for path in args.sources:
        root = path if path.is_dir() else path.parent
        for candidate in (root, root.parent / "src", root.parent):
            if candidate.is_dir() and candidate not in search_paths:
                search_paths.append(candidate)
    return scan_paths(args.sources, guards=args.guard, search_paths=search_paths)


def cmd_scan(args: argparse.Namespace) -> int:
    scanned = _scan(args)
    if args.emit is not None:
        sections = [args.emit] if args.emit else sorted(scanned)
        unknown = [s for s in sections if s not in scanned]
        if unknown:
            raise SystemExit(f"no such section: {', '.join(unknown)}")
        print("RESOURCES = {")
        for section in sections:
            print(emit_section(section, scanned[section]))
        print("}")
        return 0

    for section in sorted(scanned):
        print(f"{section}:")
        for resource in sorted(scanned[section], key=lambda r: (r.kind, r.key)):
            note = "  (bypasses cached_hub)" if resource.bypass else ""
            print(f"  - {resource}{note}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    scanned = _scan(args)
    try:
        declared = parse_declaration(args.declaration)
    except (OSError, SyntaxError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    errors, warnings = compare(scanned, declared)
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")

    if errors:
        print(
            f"\n{args.declaration} is out of sync ({len(errors)} difference(s)); "
            "`cached-hub scan --emit <section>` prints a skeleton to start from."
        )
        return 1
    total = sum(1 for items in scanned.values() for r in items if not r.bypass)
    print(f"{args.declaration} matches the sources ({total} cached loads).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cached-hub",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show the cache configuration")
    p_info.set_defaults(func=cmd_info)

    def add_from(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--from",
            dest="resources",
            metavar="MODULE:ATTR|FILE.py",
            action="append",
            required=True,
            help="resource mapping or callable returning one; a .py path is "
            "imported directly, its RESOURCES read unless :ATTR says otherwise "
            "(repeatable)",
        )

    p_list = sub.add_parser("list", help="list declared resources")
    add_from(p_list)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download declared resources")
    add_from(p_dl)
    p_dl.add_argument("--section", help="only this section (lecture, practical...)")
    p_dl.add_argument("--key", help="only the resource with this key")
    p_dl.add_argument(
        "--optional", action="store_true", help="include optional resources"
    )
    p_dl.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next --from group when a download fails",
    )
    p_dl.set_defaults(func=cmd_download)

    def add_scan(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "sources",
            nargs="+",
            type=Path,
            metavar="PATH",
            help="source files, or directories whose *.py are scanned",
        )
        p.add_argument(
            "--guard",
            action="append",
            default=list(DEFAULT_GUARDS),
            metavar="NAME",
            help="`if NAME:` marks an optional stand-in (default: %(default)s)",
        )
        p.add_argument(
            "--search-path",
            action="append",
            type=Path,
            metavar="DIR",
            help="root under which imported modules are found and scanned too",
        )

    p_scan = sub.add_parser("scan", help="list the resources the sources load")
    add_scan(p_scan)
    p_scan.add_argument(
        "--emit",
        nargs="?",
        const="",
        metavar="SECTION",
        help="print a declaration skeleton (all sections, or just SECTION)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_check = sub.add_parser(
        "check", help="check a declaration against the sources (exit 1 on drift)"
    )
    add_scan(p_check)
    p_check.add_argument(
        "--declaration",
        required=True,
        type=Path,
        metavar="FILE",
        help="Python file holding the RESOURCES mapping",
    )
    p_check.set_defaults(func=cmd_check)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Only our own logger talks at INFO: setting the root logger there as well
    # turned every HTTP request huggingface_hub makes into a line of output.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("cached_hub").setLevel(
        logging.DEBUG if args.verbose else logging.INFO
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
