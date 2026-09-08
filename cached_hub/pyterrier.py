"""PyTerrier / ir-datasets resources."""

from __future__ import annotations

from .resources import FunctionalResource, Resource


def make_pyterrier_dataset_resource(
    dataset_id: str, description: str, optional: bool = False
) -> Resource:
    """Resource that fetches a PyTerrier dataset (e.g. ``"irds:lotte/technology/dev"``).

    PyTerrier manages its own cache location (``PYTERRIER_HOME`` /
    ``IR_DATASETS_HOME``); this only triggers the download.
    """

    def download() -> str:
        import pyterrier as pt

        pt.get_dataset(dataset_id)
        return f"Downloaded {dataset_id}"

    return FunctionalResource(
        resource_type="pyterrier_dataset",
        key=dataset_id,
        description=description,
        download_fn=download,
        optional=optional,
    )
