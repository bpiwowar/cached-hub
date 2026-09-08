"""HuggingFace models, tokenizers, processors and datasets, cache first.

Loading (in notebooks)::

    from cached_hub import load_hf_model, load_hf_tokenizer, load_hf_dataset, HFModel

    tokenizer = load_hf_tokenizer("gpt2")
    model = load_hf_model("gpt2", GPT2LMHeadModel)
    dataset = load_hf_dataset("imdb", split="train")

    hf = HFModel("gpt2")        # lazy: loads on first attribute access
    hf.tokenizer, hf.model

Declaring resources to pre-download (in a course's resource list)::

    from cached_hub import make_hf_model_resource, make_hf_dataset_resource

    RESOURCES = {
        "practical1": [
            make_hf_model_resource("gpt2", model_class="AutoModelForCausalLM"),
            make_hf_tokenizer_resource("gpt2"),
            make_hf_dataset_resource("imdb", ["train", "test"]),
        ]
    }

Each loader looks in the local cache (see :mod:`cached_hub.config`) and falls
back to the Hub with a warning, or raises :class:`~cached_hub.config.CacheMissError`
in enforce mode. ``transformers`` and ``datasets`` are imported lazily.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from .config import (
    get_cache_path,
    get_hf_cache_dir,
    get_resource_path,
    is_download_complete,
    mark_download_complete,
    warn_cache_miss,
)
from .resources import FunctionalResource, Resource

logger = logging.getLogger("cached_hub")

PROVIDER = "huggingface"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _log_loaded(resource_type: str, resource_id: str, source: str) -> None:
    """Log where a resource came from; also shown inline in notebooks."""
    message = f"Loaded {resource_type} '{resource_id}' from {source}"
    logger.info(message)
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            html = f'<span style="color: #666; font-size: 0.9em;">ℹ️ {message}</span>'
            display(HTML(html))
    except (ImportError, AttributeError):
        pass


def _load_pretrained(
    resource_type: str, subdir: str, model_id: str, cls: Type, kwargs: dict
) -> Any:
    """Shared logic for models / tokenizers / processors."""
    cache_path = get_cache_path()
    local_path = get_resource_path(PROVIDER, subdir, model_id)
    if local_path is not None and is_download_complete(local_path):
        result = cls.from_pretrained(str(local_path), **kwargs)
        _log_loaded(resource_type, model_id, "local cache")
        return result

    warn_cache_miss(resource_type.capitalize(), model_id, cache_path)
    if cache_path is not None:
        # Keep the fallback download inside the shared cache
        kwargs.setdefault("cache_dir", get_hf_cache_dir())
    result = cls.from_pretrained(model_id, **kwargs)
    _log_loaded(resource_type, model_id, "HuggingFace Hub")
    return result


def load_hf_model(model_id: str, model_class: Optional[Type] = None, **kwargs) -> Any:
    """Load a model, from the local cache when present.

    Args:
        model_id: HuggingFace identifier (e.g. ``"distilbert-base-uncased"``)
        model_class: transformers class; ``AutoModel`` when ``None``
        **kwargs: forwarded to ``from_pretrained``
    """
    if model_class is None:
        from transformers import AutoModel

        model_class = AutoModel
    return _load_pretrained("model", "models", model_id, model_class, kwargs)


def load_hf_tokenizer(
    model_id: str, tokenizer_class: Optional[Type] = None, **kwargs
) -> Any:
    """Load a tokenizer, from the local cache when present (``AutoTokenizer`` by default)."""
    if tokenizer_class is None:
        from transformers import AutoTokenizer

        tokenizer_class = AutoTokenizer
    return _load_pretrained(
        "tokenizer", "tokenizers", model_id, tokenizer_class, kwargs
    )


def load_hf_processor(
    model_id: str, processor_class: Optional[Type] = None, **kwargs
) -> Any:
    """Load a processor, from the local cache when present (``AutoProcessor`` by default)."""
    if processor_class is None:
        from transformers import AutoProcessor

        processor_class = AutoProcessor
    return _load_pretrained(
        "processor", "processors", model_id, processor_class, kwargs
    )


def load_hf_dataset(
    dataset_id: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs,
) -> Any:
    """Load a dataset, from the local cache when present.

    Args:
        dataset_id: HuggingFace identifier (e.g. ``"imdb"``, ``"jxie/flickr8k"``)
        name: configuration name (e.g. ``"sst2"`` for glue)
        split: a single split; when ``None`` every cached split is returned as a
            ``DatasetDict``
        **kwargs: forwarded to ``datasets.load_dataset`` on fallback
    """
    from datasets import DatasetDict, load_dataset, load_from_disk

    cache_path = get_cache_path()
    local_base = get_resource_path(PROVIDER, "datasets", dataset_id, name)

    if local_base is not None and local_base.exists():
        if split:
            split_path = local_base / split
            if is_download_complete(split_path):
                result = load_from_disk(str(split_path))
                _log_loaded("dataset", dataset_id, "local cache")
                return result
        else:
            splits = {
                p.name: load_from_disk(str(p))
                for p in sorted(local_base.iterdir())
                if p.is_dir() and is_download_complete(p)
            }
            if splits:
                _log_loaded("dataset", dataset_id, "local cache")
                return DatasetDict(splits)

    warn_cache_miss("Dataset", dataset_id, cache_path)
    if cache_path is not None:
        kwargs.setdefault("cache_dir", get_hf_cache_dir())
    result = load_dataset(dataset_id, name, split=split, **kwargs)
    _log_loaded("dataset", dataset_id, "HuggingFace Hub")
    return result


class HFModel:
    """Lazy ``(tokenizer, model)`` pair for one model id.

    >>> hf = HFModel("gpt2")                      # AutoTokenizer / AutoModel
    >>> hf = HFModel("gpt2", GPT2Tokenizer, GPT2LMHeadModel)
    >>> hf.tokenizer   # loaded on first access, then cached
    >>> hf.model
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_cls: Optional[Type] = None,
        model_cls: Optional[Type] = None,
        **kwargs,
    ):
        """``kwargs`` are forwarded to the model's ``from_pretrained``."""
        self.model_id = model_id
        self.tokenizer_cls = tokenizer_cls
        self.model_cls = model_cls
        self.model_kwargs = kwargs

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = load_hf_tokenizer(self.model_id, self.tokenizer_cls)
        return self._tokenizer

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = load_hf_model(
                self.model_id, self.model_cls, **self.model_kwargs
            )
        return self._model


# --------------------------------------------------------------------------- #
# Resource declarations
# --------------------------------------------------------------------------- #


def _make_pretrained_resource(
    resource_type: str,
    resource_name: str,
    subdir: str,
    default_class: str,
    model_id: str,
    description: Optional[str],
    class_name: Optional[str],
    optional: bool,
) -> Resource:
    if description is None:
        description = f"HuggingFace {resource_name}: {model_id}"
    if class_name is None:
        class_name = default_class

    def download() -> str:
        import transformers

        cls = getattr(transformers, class_name)
        save_path = get_resource_path(PROVIDER, subdir, model_id)

        if save_path is None:
            # No local layout configured: just warm up the HF cache
            cls.from_pretrained(model_id, cache_dir=get_hf_cache_dir())
            return f"Cached {resource_name} {model_id} (HF cache)"

        if is_download_complete(save_path):
            return f"Already exists: {save_path}"

        save_path.mkdir(parents=True, exist_ok=True)
        cls.from_pretrained(model_id).save_pretrained(save_path)
        mark_download_complete(save_path)
        return f"Saved {resource_name} to {save_path}"

    return FunctionalResource(
        resource_type=resource_type,
        key=model_id,
        description=description,
        download_fn=download,
        optional=optional,
    )


def make_hf_model_resource(
    model_id: str,
    description: Optional[str] = None,
    model_class: Optional[str] = None,
    optional: bool = False,
) -> Resource:
    """Resource for a model; ``model_class`` is a transformers class *name*
    (default ``"AutoModel"``)."""
    return _make_pretrained_resource(
        "hf_model", "model", "models", "AutoModel",
        model_id, description, model_class, optional,
    )  # fmt: skip


def make_hf_tokenizer_resource(
    model_id: str,
    description: Optional[str] = None,
    tokenizer_class: Optional[str] = None,
    optional: bool = False,
) -> Resource:
    """Resource for a tokenizer (default class ``"AutoTokenizer"``)."""
    return _make_pretrained_resource(
        "hf_tokenizer", "tokenizer", "tokenizers", "AutoTokenizer",
        model_id, description, tokenizer_class, optional,
    )  # fmt: skip


def make_hf_processor_resource(
    model_id: str,
    description: Optional[str] = None,
    processor_class: Optional[str] = None,
    optional: bool = False,
) -> Resource:
    """Resource for a processor (default class ``"AutoProcessor"``)."""
    return _make_pretrained_resource(
        "hf_processor", "processor", "processors", "AutoProcessor",
        model_id, description, processor_class, optional,
    )  # fmt: skip


def make_hf_dataset_resource(
    dataset_id: str,
    splits: str | List[str],
    description: Optional[str] = None,
    name: Optional[str] = None,
    optional: bool = False,
) -> Resource:
    """Resource for one or more splits of a dataset.

    Splits are saved with ``save_to_disk`` under
    ``<root>/huggingface/datasets/<id>[-<name>]/<split>``.
    """
    splits_list = [splits] if isinstance(splits, str) else list(splits)
    if description is None:
        description = f"HuggingFace dataset: {dataset_id}"

    def download() -> str:
        import datasets

        save_path = get_resource_path(PROVIDER, "datasets", dataset_id, name)

        if save_path is None:
            hf_cache = get_hf_cache_dir()
            for split in splits_list:
                datasets.load_dataset(dataset_id, name, split=split, cache_dir=hf_cache)
            return f"Cached {dataset_id} splits {splits_list} (HF cache)"

        save_path.mkdir(parents=True, exist_ok=True)
        results = []
        for split in splits_list:
            split_dir = save_path / split
            if is_download_complete(split_dir):
                results.append(f"{split}: already exists")
                continue
            split_dir.mkdir(parents=True, exist_ok=True)
            dataset = datasets.load_dataset(dataset_id, name, split=split)
            dataset.save_to_disk(str(split_dir))
            mark_download_complete(split_dir)
            results.append(f"{split}: saved")
        return f"Saved to {save_path} ({', '.join(results)})"

    splits_str = "+".join(splits_list)
    key = f"{dataset_id}/{name}/{splits_str}" if name else f"{dataset_id}/{splits_str}"
    return FunctionalResource(
        resource_type="hf_dataset",
        key=key,
        description=description,
        download_fn=download,
        optional=optional,
    )
