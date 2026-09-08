"""Tests for resource declarations and selection/download helpers."""

from unittest.mock import MagicMock

from cached_hub.resources import (
    DownloadableResource,
    FunctionalResource,
    Resource,
    download_resources,
    format_resources,
    merge_resources,
    select_resources,
)


def res(key, rtype="test", optional=False, fn=None):
    return FunctionalResource(
        rtype, key, f"Resource {key}", fn or (lambda: f"done {key}"), optional
    )


def test_alias():
    assert DownloadableResource is Resource


def test_functional_resource_properties():
    fn = MagicMock(return_value="Download complete")
    r = FunctionalResource("test_type", "test-key", "Test description", fn)
    assert r.resource_type == "test_type"
    assert r.key == "test-key"
    assert r.description == "Test description"
    assert r.optional is False
    assert r.download() == "Download complete"
    fn.assert_called_once()
    assert "test_type:test-key" in repr(r)


def test_equality_by_type_and_key():
    assert res("a") == res("a")
    assert hash(res("a")) == hash(res("a"))
    assert res("a") != res("b")
    assert res("a", "hf_model") != res("a", "hf_dataset")
    assert res("a") != "a"


def test_select_deduplicates_across_sections():
    shared = res("shared")
    resources = {"l1": [shared, res("x")], "l2": [res("shared"), res("y")]}
    selected = select_resources(resources)
    assert [(s, r.key) for s, r in selected] == [
        ("l1", "shared"),
        ("l1", "x"),
        ("l2", "y"),
    ]


def test_select_optional():
    resources = {"l1": [res("a"), res("opt", optional=True)]}
    assert [r.key for _, r in select_resources(resources)] == ["a"]
    assert [r.key for _, r in select_resources(resources, include_optional=True)] == [
        "a",
        "opt",
    ]
    # --key targets optional resources too
    assert [r.key for _, r in select_resources(resources, key="opt")] == ["opt"]


def test_select_section():
    resources = {"l1": [res("a")], "l2": [res("b")]}
    assert [r.key for _, r in select_resources(resources, section="l2")] == ["b"]
    assert select_resources(resources, section="nope") == []


def test_download_resources_calls_each_once():
    calls = []

    def track(name):
        def fn():
            calls.append(name)
            return f"Downloaded {name}"

        return fn

    shared = res("shared", fn=track("shared"))
    resources = {"l1": [shared, res("a", fn=track("a"))], "l2": [shared]}
    results = download_resources(resources)
    assert calls == ["shared", "a"]
    assert [status for _, _, status in results] == [
        "Downloaded shared",
        "Downloaded a",
    ]


def test_format_resources_sorted_with_optional_marker():
    resources = {"b": [res("y", optional=True)], "a": [res("x")]}
    assert format_resources(resources) == [
        "a:",
        "  - x: Resource x",
        "b:",
        "  - y: Resource y (optional)",
    ]
    assert format_resources(resources, indent="  ")[0] == "  a:"


def test_merge_resources():
    merged = merge_resources({"a": [res("1")]}, {"a": [res("2")], "b": [res("3")]})
    assert [r.key for r in merged["a"]] == ["1", "2"]
    assert [r.key for r in merged["b"]] == ["3"]
