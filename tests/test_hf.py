"""Tests for the HuggingFace loaders and resource factories."""

import os
from unittest.mock import MagicMock, patch

import pytest

from cached_hub.config import ENV_ENFORCE, ENV_PATH, CacheMissError
from cached_hub.hf import (
    HFModel,
    load_hf_dataset,
    load_hf_model,
    load_hf_tokenizer,
    make_hf_dataset_resource,
    make_hf_model_resource,
    make_hf_processor_resource,
    make_hf_tokenizer_resource,
)


def _cached(tmp_path, *parts):
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True)
    (path / ".downloaded.ok").touch()
    return path


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def test_load_model_from_cache(tmp_path):
    local = _cached(tmp_path, "huggingface", "models", "org-model")
    cls = MagicMock()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        load_hf_model("org/model", cls, torch_dtype="auto")
    cls.from_pretrained.assert_called_once_with(str(local), torch_dtype="auto")


def test_load_model_falls_back_to_hub_inside_cache(tmp_path, caplog):
    cls = MagicMock()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with caplog.at_level("WARNING", logger="cached_hub"):
            load_hf_model("org/model", cls)
    cls.from_pretrained.assert_called_once_with(
        "org/model", cache_dir=str(tmp_path / "huggingface")
    )
    assert "not found in local cache" in caplog.text


def test_load_model_without_cache_uses_hub_defaults():
    cls = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        load_hf_model("org/model", cls)
    cls.from_pretrained.assert_called_once_with("org/model")


def test_load_model_enforce_mode_raises(tmp_path):
    cls = MagicMock()
    env = {ENV_PATH: str(tmp_path), ENV_ENFORCE: "1"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(CacheMissError):
            load_hf_model("org/model", cls)
    cls.from_pretrained.assert_not_called()


def test_load_model_default_class_is_automodel():
    transformers = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict("sys.modules", {"transformers": transformers}):
            load_hf_model("gpt2")
    transformers.AutoModel.from_pretrained.assert_called_once_with("gpt2")


def test_load_tokenizer_from_cache(tmp_path):
    local = _cached(tmp_path, "huggingface", "tokenizers", "gpt2")
    cls = MagicMock()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        load_hf_tokenizer("gpt2", cls)
    cls.from_pretrained.assert_called_once_with(str(local))


def test_load_dataset_single_split_from_cache(tmp_path):
    split_dir = _cached(tmp_path, "huggingface", "datasets", "imdb", "train")
    datasets = MagicMock()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch.dict("sys.modules", {"datasets": datasets}):
            load_hf_dataset("imdb", split="train")
    datasets.load_from_disk.assert_called_once_with(str(split_dir))
    datasets.load_dataset.assert_not_called()


def test_load_dataset_all_splits_from_cache(tmp_path):
    _cached(tmp_path, "huggingface", "datasets", "glue-sst2", "train")
    _cached(tmp_path, "huggingface", "datasets", "glue-sst2", "validation")
    (tmp_path / "huggingface" / "datasets" / "glue-sst2" / "partial").mkdir()
    datasets = MagicMock()
    datasets.load_from_disk.side_effect = lambda p: f"ds:{os.path.basename(p)}"
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch.dict("sys.modules", {"datasets": datasets}):
            load_hf_dataset("glue", name="sst2")
    datasets.DatasetDict.assert_called_once_with(
        {"train": "ds:train", "validation": "ds:validation"}
    )


def test_load_dataset_fallback(tmp_path):
    datasets = MagicMock()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch.dict("sys.modules", {"datasets": datasets}):
            load_hf_dataset("imdb", split="test")
    datasets.load_dataset.assert_called_once_with(
        "imdb", None, split="test", cache_dir=str(tmp_path / "huggingface")
    )


def test_hfmodel_is_lazy_and_cached():
    tok_cls, model_cls = MagicMock(), MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        hf = HFModel("gpt2", tok_cls, model_cls, device_map="auto")
        tok_cls.from_pretrained.assert_not_called()
        assert hf.tokenizer is hf.tokenizer
        assert hf.model is hf.model
    tok_cls.from_pretrained.assert_called_once_with("gpt2")
    model_cls.from_pretrained.assert_called_once_with("gpt2", device_map="auto")


# --------------------------------------------------------------------------- #
# Resource factories
# --------------------------------------------------------------------------- #

TRANSFORMERS_RESOURCES = [
    (make_hf_model_resource, "model", "models", "AutoModel"),
    (make_hf_tokenizer_resource, "tokenizer", "tokenizers", "AutoTokenizer"),
    (make_hf_processor_resource, "processor", "processors", "AutoProcessor"),
]


@pytest.mark.parametrize("make_fn,kind,subdir,default_class", TRANSFORMERS_RESOURCES)
class TestPretrainedResources:
    def test_key_and_description(self, make_fn, kind, subdir, default_class):
        r = make_fn("bert-base-uncased")
        assert r.key == "bert-base-uncased"
        assert r.description == f"HuggingFace {kind}: bert-base-uncased"
        assert make_fn("x", "Custom").description == "Custom"
        assert make_fn("x", optional=True).optional is True

    def test_skips_when_cached(self, make_fn, kind, subdir, default_class, tmp_path):
        _cached(tmp_path, "huggingface", subdir, "bert-base-uncased")
        transformers = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            with patch.dict("sys.modules", {"transformers": transformers}):
                result = make_fn("bert-base-uncased").download()
        assert "Already exists" in result
        getattr(transformers, default_class).from_pretrained.assert_not_called()

    def test_saves_into_cache(self, make_fn, kind, subdir, default_class, tmp_path):
        transformers = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            with patch.dict("sys.modules", {"transformers": transformers}):
                result = make_fn("org/bert").download()
        target = tmp_path / "huggingface" / subdir / "org-bert"
        assert "Saved" in result
        cls = getattr(transformers, default_class)
        cls.from_pretrained.assert_called_once_with("org/bert")
        cls.from_pretrained.return_value.save_pretrained.assert_called_once_with(target)
        assert (target / ".downloaded.ok").exists()

    def test_warms_hf_cache_without_root(self, make_fn, kind, subdir, default_class):
        transformers = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"transformers": transformers}):
                result = make_fn("bert-base-uncased").download()
        assert "Cached" in result
        getattr(transformers, default_class).from_pretrained.assert_called_once_with(
            "bert-base-uncased", cache_dir=None
        )


def test_model_resource_custom_class():
    transformers = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict("sys.modules", {"transformers": transformers}):
            make_hf_model_resource(
                "gpt2", model_class="AutoModelForCausalLM"
            ).download()
    transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


class TestDatasetResource:
    def test_keys(self):
        assert make_hf_dataset_resource("imdb", "train").key == "imdb/train"
        assert (
            make_hf_dataset_resource("imdb", ["train", "test"]).key == "imdb/train+test"
        )
        assert (
            make_hf_dataset_resource("glue", "validation", name="sst2").key
            == "glue/sst2/validation"
        )

    def test_description_and_optional(self):
        assert make_hf_dataset_resource("imdb", "train").description == (
            "HuggingFace dataset: imdb"
        )
        assert make_hf_dataset_resource("imdb", "train", "IMDB").description == "IMDB"
        assert make_hf_dataset_resource("imdb", "train", optional=True).optional

    def test_skips_cached_split(self, tmp_path):
        _cached(tmp_path, "huggingface", "datasets", "imdb", "train")
        datasets = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            with patch.dict("sys.modules", {"datasets": datasets}):
                result = make_hf_dataset_resource("imdb", "train").download()
        assert "already exists" in result
        datasets.load_dataset.assert_not_called()

    def test_saves_missing_splits(self, tmp_path):
        _cached(tmp_path, "huggingface", "datasets", "imdb", "train")
        datasets = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            with patch.dict("sys.modules", {"datasets": datasets}):
                result = make_hf_dataset_resource("imdb", ["train", "test"]).download()
        assert "train: already exists" in result and "test: saved" in result
        datasets.load_dataset.assert_called_once_with("imdb", None, split="test")
        test_dir = tmp_path / "huggingface" / "datasets" / "imdb" / "test"
        datasets.load_dataset.return_value.save_to_disk.assert_called_once_with(
            str(test_dir)
        )
        assert (test_dir / ".downloaded.ok").exists()

    def test_warms_hf_cache_without_root(self):
        datasets = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"datasets": datasets}):
                result = make_hf_dataset_resource("imdb", "train").download()
        assert "Cached" in result
        datasets.load_dataset.assert_called_once_with(
            "imdb", None, split="train", cache_dir=None
        )
