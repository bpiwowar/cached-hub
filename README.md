# cached-hub

**Load HuggingFace models and datasets from a shared local cache, fall back to
the Hub, and pre-populate that cache from a declarative list of what a course
needs.**

## Why

In a classroom, every student pulling `gpt2`, `SmolLM2` and `imdb` from the Hub
at the same minute is slow, fragile, and sometimes impossible (no home directory
quota, shaky proxy, offline lab). The usual answer is a shared read-only
directory pre-filled by the instructor. `cached-hub` makes that directory a
first-class thing:

- notebook code calls `load_hf_model("gpt2")` and gets the cached copy when it
  exists, the Hub otherwise, with a one-line note saying which;
- the instructor declares the resources once, and `cached-hub download` fills
  the cache (each shared resource once, optional ones on demand);
- an *enforce* mode turns any cache miss into an error, so you can check that
  every notebook runs fully from the cache before the session.

The package has **no required dependency**: `transformers`, `datasets`,
`pyterrier` and `datamaestro` are imported only by the functions that use them.
It was extracted from the Sorbonne Master MIND `master-mind` tool so that a
course only needs this package (plus
[jupytext-notebook-helper](https://github.com/bpiwowar/jupytext-notebook-helper)
for building notebooks).

## Install

```sh
pip install cached-hub            # loaders only (bring your own transformers/datasets)
pip install "cached-hub[hf]"      # + transformers, datasets
```

## In notebooks

```python
from cached_hub import load_hf_model, load_hf_tokenizer, load_hf_dataset, HFModel
from transformers import AutoModelForCausalLM

tokenizer = load_hf_tokenizer("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = load_hf_model("HuggingFaceTB/SmolLM2-1.7B-Instruct", AutoModelForCausalLM, device_map="auto")
train = load_hf_dataset("imdb", split="train")
sst2 = load_hf_dataset("glue", name="sst2")          # DatasetDict of the cached splits

hf = HFModel("gpt2")            # lazy: nothing is loaded yet
hf.tokenizer, hf.model          # AutoTokenizer / AutoModel, loaded on first access
```

Each loader checks the local cache first, then falls back to the Hub with a
warning. Extra keyword arguments go to `from_pretrained` / `load_dataset`.

## Configuration

| Variable             | Effect                                                                 |
|----------------------|------------------------------------------------------------------------|
| `CACHED_HUB_PATH`    | Root of the shared cache. Unset: the library does nothing (see below). |
| `CACHED_HUB_ENFORCE` | If set (any value), a cache miss raises `CacheMissError` instead of falling back. |

Layout under the root (`org/name` becomes `org-name`):

```
$CACHED_HUB_PATH/huggingface/models/<id>/                 model.save_pretrained()
$CACHED_HUB_PATH/huggingface/tokenizers/<id>/
$CACHED_HUB_PATH/huggingface/processors/<id>/
$CACHED_HUB_PATH/huggingface/datasets/<id>[-<name>]/<split>/   dataset.save_to_disk()
$CACHED_HUB_PATH/huggingface/                             HF cache_dir used for fallbacks
```

A directory is used only when it contains the marker `.downloaded.ok`, written
after a successful download, so a half-copied model is never picked up.

**Without `CACHED_HUB_PATH`, `cached-hub` adds no caching of its own.**
`load_hf_model("gpt2", cls, **kw)` is then exactly `cls.from_pretrained("gpt2", **kw)`,
and `load_hf_dataset(...)` exactly `datasets.load_dataset(...)`: the usual
HuggingFace cache (`~/.cache/huggingface`, `HF_HOME`) applies as it always does,
and `cached-hub download` merely warms it. Notebooks can therefore import from
`cached_hub` unconditionally and run unchanged on a laptop or on Colab; only the
classroom machines set the variable.

## Declaring and downloading resources

A course lists what it needs as `{section: [resources]}`:

```python
# mycourse/resources.py
from cached_hub import (
    make_hf_model_resource, make_hf_tokenizer_resource, make_hf_processor_resource,
    make_hf_dataset_resource, make_pyterrier_dataset_resource, make_datamaestro_resource,
)

RESOURCES = {
    "practical1": [
        make_hf_model_resource("gpt2", model_class="GPT2LMHeadModel"),
        make_hf_tokenizer_resource("gpt2", tokenizer_class="GPT2Tokenizer"),
        make_hf_dataset_resource("imdb", ["train", "test"]),
    ],
    "practical2": [
        make_hf_model_resource("Qwen/Qwen2.5-7B-Instruct", model_class="AutoModelForCausalLM", optional=True),
        make_pyterrier_dataset_resource("irds:lotte/technology/dev/search", "LoTTE technology"),
    ],
}
```

then, on the machine that hosts the cache:

```sh
export CACHED_HUB_PATH=/shared/cache
cached-hub info
cached-hub list     --from mycourse.resources:RESOURCES
cached-hub download --from mycourse.resources:RESOURCES               # everything but optional
cached-hub download --from mycourse.resources:RESOURCES --section practical2 --optional
cached-hub download --from mycourse.resources:RESOURCES --key gpt2
```

`--from MODULE:ATTR` imports `MODULE` and reads `ATTR` from it: a
`{section: [resources]}` mapping, or a zero-argument callable returning one
(dotted attributes such as `plugin.Course.resources` are followed). No source
scanning is involved; `RESOURCES` above is only a naming convention. The option
can be repeated. Resources are identified by `(type, key)`, so a model shared
by several practicals is downloaded once. `HF_HUB_OFFLINE` is lifted for the
duration of a download.

The same helpers are available from Python (`download_resources`,
`select_resources`, `format_resources`, `merge_resources`), and any object with
`resource_type`, `key`, `description`, `optional` and `download()` is a valid
resource (`FunctionalResource` wraps a plain function).

## Checking a cache before class

```sh
CACHED_HUB_PATH=/shared/cache CACHED_HUB_ENFORCE=1 python practical1.py
```

fails at the first resource that would have gone to the Hub.
