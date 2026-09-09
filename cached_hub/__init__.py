"""cached-hub: HuggingFace (and other) resources from a shared local cache.

Notebook side::

    from cached_hub import load_hf_model, load_hf_tokenizer, load_hf_dataset, HFModel

Course side (declare what to pre-download)::

    from cached_hub import make_hf_model_resource, make_hf_dataset_resource
    RESOURCES = {"practical1": [make_hf_model_resource("gpt2"), ...]}

then ``cached-hub download --from mycourse.resources:RESOURCES``.

``cached-hub scan sources/`` reads the loader calls back out of the sources, so
that a declaration can be written from them (``--emit``) and kept honest
(``cached-hub check sources/ --declaration mycourse/resources.py``).

Configure the cache root with ``CACHED_HUB_PATH`` (see :mod:`cached_hub.config`).
"""

from .config import (
    ENV_ENFORCE,
    ENV_PATH,
    CacheMissError,
    get_cache_path,
    is_enforce_mode,
)
from .datamaestro import make_datamaestro_resource
from .hf import (
    HFModel,
    load_hf_dataset,
    load_hf_model,
    load_hf_processor,
    load_hf_tokenizer,
    make_hf_dataset_resource,
    make_hf_model_resource,
    make_hf_processor_resource,
    make_hf_tokenizer_resource,
)
from .pyterrier import make_pyterrier_dataset_resource
from .resources import (
    DownloadableResource,
    FunctionalResource,
    Resource,
    Resources,
    download_resources,
    format_resources,
    merge_resources,
    select_resources,
)
from .scan import (
    ScannedResource,
    compare,
    emit_section,
    parse_declaration,
    scan_paths,
)

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - not installed from a build
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # config
    "ENV_PATH",
    "ENV_ENFORCE",
    "CacheMissError",
    "get_cache_path",
    "is_enforce_mode",
    # loading
    "HFModel",
    "load_hf_dataset",
    "load_hf_model",
    "load_hf_processor",
    "load_hf_tokenizer",
    # resources
    "Resource",
    "DownloadableResource",
    "FunctionalResource",
    "Resources",
    "download_resources",
    "format_resources",
    "merge_resources",
    "select_resources",
    "make_hf_dataset_resource",
    "make_hf_model_resource",
    "make_hf_processor_resource",
    "make_hf_tokenizer_resource",
    "make_pyterrier_dataset_resource",
    "make_datamaestro_resource",
    # scanning the sources that load them
    "ScannedResource",
    "compare",
    "emit_section",
    "parse_declaration",
    "scan_paths",
]
