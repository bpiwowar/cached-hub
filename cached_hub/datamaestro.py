"""datamaestro resources."""

from __future__ import annotations

from typing import Optional

from .resources import FunctionalResource, Resource


def make_datamaestro_resource(
    dataset_id: str, description: Optional[str] = None, optional: bool = False
) -> Resource:
    """Resource that prepares a datamaestro dataset (e.g. ``"com.lecun.mnist"``).

    datamaestro manages its own storage (``DATAMAESTRO_DIR``); this only
    triggers ``prepare_dataset``.
    """
    if description is None:
        description = f"datamaestro dataset: {dataset_id}"

    def download() -> str:
        from datamaestro import prepare_dataset

        prepare_dataset(dataset_id)
        return f"Prepared {dataset_id}"

    return FunctionalResource(
        resource_type="datamaestro",
        key=dataset_id,
        description=description,
        download_fn=download,
        optional=optional,
    )
