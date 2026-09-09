"""Tests for reading resources out of the sources that load them."""

import textwrap
from pathlib import Path

import pytest

from cached_hub.cli import main
from cached_hub.scan import compare, emit_section, parse_declaration, scan_paths

PRACTICAL1 = '''
"""A course notebook."""
from cached_hub import load_hf_dataset, load_hf_model, load_hf_tokenizer

MODEL = "big/model"
if test_mode:
    MODEL = "small/model"

data = load_hf_dataset("some/corpus", "config-a", split="train")
tokenizer = load_hf_tokenizer(MODEL)
model = load_hf_model(MODEL, AutoModelForCausalLM)
'''

PRACTICAL2 = """
from cached_hub import HFModel
from mycourse.helpers import prepare

hf = HFModel("other/model", model_cls=AutoModelForCausalLM)
index = pt.get_dataset("irds:lotte/technology/dev/search")
reward = AutoModelForSequenceClassification.from_pretrained("bypassed/model")
"""

HELPER = """
from cached_hub import load_hf_processor

processor = load_hf_processor("helper/model", CLIPProcessor)
"""

DECLARATION = '''
"""Declared resources."""
from cached_hub import (
    make_hf_dataset_resource,
    make_hf_model_resource,
    make_hf_processor_resource,
    make_hf_tokenizer_resource,
    make_pyterrier_dataset_resource,
)

BIG = "big/model"

RESOURCES = {
    "practical1": [
        make_hf_dataset_resource("some/corpus", ["train"], "Corpus", name="config-a"),
        make_hf_tokenizer_resource(BIG, "Tokenizer"),
        make_hf_model_resource(BIG, "Model", model_class="AutoModelForCausalLM"),
        make_hf_tokenizer_resource("small/model", "Small", optional=True),
        make_hf_model_resource(
            "small/model", "Small", model_class="AutoModelForCausalLM", optional=True
        ),
    ],
    "practical2": [
        make_hf_model_resource(
            "other/model", "Other", model_class="AutoModelForCausalLM"
        ),
        make_hf_tokenizer_resource("other/model", "Other"),
        make_hf_processor_resource(
            "helper/model", "Helper", processor_class="CLIPProcessor"
        ),
        make_pyterrier_dataset_resource("irds:lotte/technology/dev/search", "LoTTE"),
    ],
}
'''


@pytest.fixture
def course(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "practical1.py").write_text(textwrap.dedent(PRACTICAL1))
    (sources / "practical2.py").write_text(textwrap.dedent(PRACTICAL2))
    (sources / "_private.py").write_text("load_hf_model('never/scanned')\n")
    helpers = tmp_path / "src" / "mycourse"
    helpers.mkdir(parents=True)
    (helpers / "__init__.py").write_text("")
    (helpers / "helpers.py").write_text(textwrap.dedent(HELPER))
    (tmp_path / "declaration.py").write_text(textwrap.dedent(DECLARATION))
    return tmp_path


def scan(course: Path):
    return scan_paths([course / "sources"], search_paths=[course / "src"])


def keys(resources, kind=None):
    return {r.key for r in resources if kind is None or r.kind == kind}


def test_sections_are_file_stems_and_skip_private(course):
    assert set(scan(course)) == {"practical1", "practical2"}


def test_constants_and_guarded_variants(course):
    practical1 = scan(course)["practical1"]
    assert keys(practical1, "model") == {"big/model", "small/model"}
    optional = {r.key: r.optional for r in practical1 if r.kind == "model"}
    assert optional == {"big/model": False, "small/model": True}


def test_dataset_second_argument_is_a_config_not_a_split(course):
    (dataset,) = [r for r in scan(course)["practical1"] if r.kind == "dataset"]
    assert dataset.key == "some/corpus[config-a]"
    assert dataset.splits == ("train",)


def test_hf_model_yields_model_and_tokenizer(course):
    practical2 = scan(course)["practical2"]
    assert ("model", "other/model") in {r.identity for r in practical2}
    assert ("tokenizer", "other/model") in {r.identity for r in practical2}


def test_imported_helpers_are_scanned_with_their_importer(course):
    practical2 = scan(course)["practical2"]
    (processor,) = [r for r in practical2 if r.kind == "processor"]
    assert (processor.key, processor.cls) == ("helper/model", "CLIPProcessor")


def test_plain_from_pretrained_is_a_bypass(course):
    practical2 = scan(course)["practical2"]
    (bypass,) = [r for r in practical2 if r.bypass]
    assert bypass.key == "bypassed/model"


def test_declaration_is_parsed_not_imported(course):
    declared = parse_declaration(course / "declaration.py")
    assert set(declared) == {"practical1", "practical2"}
    model = next(
        r for r in declared["practical1"] if r.identity == ("model", "big/model")
    )
    assert model.cls == "AutoModelForCausalLM"
    tokenizer = next(
        r for r in declared["practical1"] if r.identity == ("tokenizer", "big/model")
    )
    assert tokenizer.cls is None  # left to the default


def test_declaration_in_sync(course):
    errors, warnings = compare(
        scan(course), parse_declaration(course / "declaration.py")
    )
    assert (errors, warnings) == ([], [])


def test_missing_declaration_is_an_error(course):
    declaration = course / "declaration.py"
    declaration.write_text(
        declaration.read_text().replace(
            '        make_hf_tokenizer_resource(BIG, "Tokenizer"),\n', ""
        )
    )
    errors, _ = compare(scan(course), parse_declaration(declaration))
    assert errors == [
        "practical1: loaded but not declared — tokenizer big/model (AutoTokenizer)"
    ]


def test_stale_declaration_is_an_error(course):
    declaration = course / "declaration.py"
    declaration.write_text(
        declaration.read_text().replace(
            '    "practical2": [',
            '    "practical2": [\n        make_hf_model_resource("gone/model", "Stale"),',
        )
    )
    errors, _ = compare(scan(course), parse_declaration(declaration))
    assert errors == ["practical2: declared but never loaded — model gone/model"]


def test_class_and_optional_mismatches_are_warnings(course):
    declaration = course / "declaration.py"
    declaration.write_text(
        declaration.read_text()
        .replace('processor_class="CLIPProcessor"', 'processor_class="AutoProcessor"')
        .replace(
            'make_hf_tokenizer_resource("small/model", "Small", optional=True)',
            'make_hf_tokenizer_resource("small/model", "Small")',
        )
    )
    errors, warnings = compare(scan(course), parse_declaration(declaration))
    assert errors == []
    assert warnings == [
        "practical1: small/model declared required but loaded only under a guard",
        "practical2: helper/model declared as AutoProcessor, loaded as CLIPProcessor",
    ]


def test_emit_round_trips_through_the_parser(course, tmp_path):
    scanned = scan(course)
    body = "\n".join(emit_section(name, scanned[name]) for name in sorted(scanned))
    generated = tmp_path / "generated.py"
    generated.write_text(f"RESOURCES = {{\n{body}\n}}\n")

    declared = parse_declaration(generated)
    errors, warnings = compare(scanned, declared)
    assert errors == []
    # Splits are guessed by `emit`, never compared, so only they may differ.
    assert warnings == []


def test_cli_check_reports_drift(course, capsys):
    argv = [
        "check",
        str(course / "sources"),
        "--declaration",
        str(course / "declaration.py"),
    ]
    assert main(argv) == 0
    assert "matches the sources" in capsys.readouterr().out

    declaration = course / "declaration.py"
    declaration.write_text(
        declaration.read_text().replace(
            '        make_hf_tokenizer_resource(BIG, "Tokenizer"),\n', ""
        )
    )
    assert main(argv) == 1
    assert "loaded but not declared" in capsys.readouterr().out


def test_cli_scan_emit_is_valid_python(course, capsys):
    assert main(["scan", str(course / "sources"), "--emit", "practical1"]) == 0
    compile(capsys.readouterr().out, "<emit>", "exec")


def test_cli_scan_unknown_section(course):
    with pytest.raises(SystemExit):
        main(["scan", str(course / "sources"), "--emit", "nope"])
