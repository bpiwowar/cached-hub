"""Declarative resources: what a course needs downloaded, and how to do it.

A course describes its resources as a mapping from a section name (a lecture,
a practical, ...) to a list of :class:`Resource` objects::

    RESOURCES = {
        "practical1": [make_hf_model_resource("gpt2"), make_hf_tokenizer_resource("gpt2")],
        "practical2": [make_hf_dataset_resource("imdb", ["train", "test"])],
    }

Two resources are the *same* when they share ``(resource_type, key)``; the
download helpers use that to fetch shared resources only once.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("cached_hub")


class Resource(ABC):
    """A downloadable resource (model, dataset, ...).

    Subclasses implement ``resource_type`` and ``key`` so that equality and
    hashing identify duplicates across sections.
    """

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """Kind of resource, e.g. ``"hf_model"``, ``"hf_dataset"``."""

    @property
    @abstractmethod
    def key(self) -> str:
        """Identifier, unique within ``resource_type``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human description."""

    @property
    def optional(self) -> bool:
        """Optional resources are skipped unless explicitly requested."""
        return False

    @abstractmethod
    def download(self) -> str:
        """Fetch the resource into the cache; return a short status message."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resource):
            return NotImplemented
        return (self.resource_type, self.key) == (other.resource_type, other.key)

    def __hash__(self) -> int:
        return hash((self.resource_type, self.key))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.resource_type}:{self.key}>"


#: Historical name, kept for master-mind plugins.
DownloadableResource = Resource

#: Resources grouped by section name.
Resources = Dict[str, List[Resource]]


class FunctionalResource(Resource):
    """A resource whose download is a plain function."""

    def __init__(
        self,
        resource_type: str,
        key: str,
        description: str,
        download_fn: Callable[[], str],
        optional: bool = False,
    ) -> None:
        self._resource_type = resource_type
        self._key = key
        self._description = description
        self._download_fn = download_fn
        self._optional = optional

    @property
    def resource_type(self) -> str:
        return self._resource_type

    @property
    def key(self) -> str:
        return self._key

    @property
    def description(self) -> str:
        return self._description

    @property
    def optional(self) -> bool:
        return self._optional

    def download(self) -> str:
        return self._download_fn()


def select_resources(
    resources: Resources,
    *,
    section: Optional[str] = None,
    key: Optional[str] = None,
    include_optional: bool = False,
) -> List[Tuple[str, Resource]]:
    """Flatten ``resources`` into ``(section, resource)`` pairs, de-duplicated.

    Args:
        section: keep only this section.
        key: keep only the resource(s) with this key (optional ones included).
        include_optional: keep optional resources (ignored when ``key`` is set).
    """
    selected: List[Tuple[str, Resource]] = []
    seen: set = set()
    for section_name, items in resources.items():
        if section is not None and section_name != section:
            continue
        for resource in items:
            if key is not None:
                if resource.key != key:
                    continue
            elif resource.optional and not include_optional:
                logger.info("Skipping %s/%s (optional)", section_name, resource.key)
                continue
            if resource in seen:
                logger.info(
                    "Skipping %s/%s (already selected)", section_name, resource.key
                )
                continue
            seen.add(resource)
            selected.append((section_name, resource))
    return selected


def download_resources(
    resources: Resources,
    *,
    section: Optional[str] = None,
    key: Optional[str] = None,
    include_optional: bool = False,
    keep_going: bool = False,
    failures: Optional[List[Tuple[str, Resource, Exception]]] = None,
) -> List[Tuple[str, Resource, str]]:
    """Download the selected resources (see :func:`select_resources`).

    Returns ``(section, resource, status)`` triples, one per resource that was
    downloaded. Exceptions from ``download()`` propagate, unless ``keep_going``
    is set: the remaining resources are then still attempted, and each failure
    is logged and appended to ``failures`` as ``(section, resource, exception)``
    — filling a cache for a classroom should not stop at the one model whose
    extra dependency is missing.
    """
    results = []
    for section_name, resource in select_resources(
        resources, section=section, key=key, include_optional=include_optional
    ):
        logger.info(
            "Downloading %s/%s: %s", section_name, resource.key, resource.description
        )
        try:
            status = resource.download()
        except Exception as exc:
            if not keep_going:
                raise
            logger.error("  -> FAILED %s/%s: %s", section_name, resource.key, exc)
            logger.debug("traceback", exc_info=True)
            if failures is not None:
                failures.append((section_name, resource, exc))
            continue
        if status:
            logger.info("  -> %s", status)
        results.append((section_name, resource, status))
    return results


def format_resources(resources: Resources, indent: str = "") -> List[str]:
    """Render ``resources`` as indented text lines (sections sorted)."""
    lines = []
    for section_name in sorted(resources):
        lines.append(f"{indent}{section_name}:")
        for resource in resources[section_name]:
            marker = " (optional)" if resource.optional else ""
            lines.append(f"{indent}  - {resource.key}: {resource.description}{marker}")
    return lines


def merge_resources(*groups: Resources) -> Resources:
    """Merge several resource mappings (same-named sections are concatenated)."""
    merged: Resources = {}
    for group in groups:
        for section_name, items in group.items():
            merged.setdefault(section_name, []).extend(items)
    return merged


def iter_all(resources: Resources) -> Iterable[Resource]:
    """Iterate over every resource, in section order, without de-duplication."""
    for items in resources.values():
        yield from items
