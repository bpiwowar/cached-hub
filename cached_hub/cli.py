"""Command line: inspect the cache, list and download declared resources.

Resources are given with ``--from module:attribute``, where the attribute is a
resource mapping (``{section: [Resource, ...]}``) or a zero-argument callable
returning one::

    cached-hub info
    cached-hub list --from llm_course.resources:RESOURCES
    cached-hub download --from llm_course.resources:get_resources --section practical2
    cached-hub download --from llm_course.resources:RESOURCES --key gpt2
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import List, Optional, Sequence, Tuple

from . import config
from .resources import Resources, download_resources, format_resources


def load_resources(spec: str) -> Resources:
    """Resolve ``module:attribute`` to a resource mapping."""
    module_name, sep, attr_path = spec.partition(":")
    if not sep or not attr_path:
        raise SystemExit(f"invalid resource spec {spec!r}: expected module:attribute")
    obj = importlib.import_module(module_name)
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

    failures = 0
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
                )
            except Exception:
                if not args.keep_going:
                    raise
                failures += 1
                logging.exception("%s: download failed", spec)
    return 1 if failures else 0


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
            metavar="MODULE:ATTR",
            action="append",
            required=True,
            help="resource mapping or callable returning one (repeatable)",
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
