"""Tests for the generic URL/archive cache (cached_download)."""

import os
import tarfile
from unittest.mock import patch

import pytest

from cached_hub.config import ENV_ENFORCE, ENV_PATH, CacheMissError
from cached_hub.web import cached_download

URL = "https://example.org/data/thing.tar.gz"
FILE_URL = "https://example.org/data/thing.json"


def _make_tarball(path, content="hello"):
    src = path.parent / f"{path.name}-src"
    src.mkdir(parents=True)
    (src / "data.txt").write_text(content)
    with tarfile.open(path, "w:gz") as tar:
        tar.add(src / "data.txt", arcname="data.txt")
    return path


def _fake_urlretrieve(dest_content="hello"):
    def _retrieve(url, dest):
        if url.endswith(".tar.gz"):
            _make_tarball(dest, dest_content)
        else:
            dest.write_text(dest_content)

    return _retrieve


def test_no_cache_downloads_to_outputs_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve()):
            path = cached_download(URL, "llm/thing")
    assert path.resolve() == tmp_path / "outputs" / "cache" / "llm" / "thing"
    assert (path / "data.txt").read_text() == "hello"


def test_no_cache_reuses_local_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "outputs" / "cache" / "llm" / "thing"
    dest.mkdir(parents=True)
    (dest / "data.txt").write_text("already there")
    (dest / ".downloaded.ok").touch()
    with patch.dict(os.environ, {}, clear=True):
        with patch("cached_hub.web.urlretrieve") as urlretrieve:
            path = cached_download(URL, "llm/thing")
    urlretrieve.assert_not_called()
    assert path.resolve() == dest


def test_writable_root_downloads_and_marks_complete(tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve()):
            path = cached_download(URL, "llm/thing")
    assert path == tmp_path / "custom" / "llm" / "thing"
    assert (path / ".downloaded.ok").exists()
    assert (path / "data.txt").read_text() == "hello"


def test_writable_root_reuses_completed_entry(tmp_path):
    dest = tmp_path / "custom" / "llm" / "thing"
    dest.mkdir(parents=True)
    (dest / ".downloaded.ok").touch()
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch("cached_hub.web.urlretrieve") as urlretrieve:
            path = cached_download(URL, "llm/thing")
    urlretrieve.assert_not_called()
    assert path == dest


def test_reuses_local_archive_without_redownloading_or_deleting(tmp_path):
    """A tarball a course already built locally (e.g. via its own `make`
    target) must be extracted as is, never re-fetched from the URL, and must
    survive the extraction -- other tooling (publishing it) may still need
    it."""
    dest = tmp_path / "custom" / "llm" / "thing"
    archive = dest.with_suffix(".tar.gz")
    _make_tarball(archive, "built locally")
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch("cached_hub.web.urlretrieve") as urlretrieve:
            path = cached_download(URL, "llm/thing")
    urlretrieve.assert_not_called()
    assert (path / "data.txt").read_text() == "built locally"
    assert archive.exists()  # not deleted: we did not download it


def test_readonly_root_falls_back_to_temp(tmp_path, monkeypatch):
    monkeypatch.setattr("cached_hub.web._is_writable", lambda directory: False)
    monkeypatch.setattr("getpass.getuser", lambda: "test-user")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path / "readonly")}, clear=True):
        with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve()):
            path = cached_download(URL, "llm/thing")
    assert path == tmp_path / "tmp" / "test-user" / "cached-hub" / "llm" / "thing"
    assert (path / "data.txt").read_text() == "hello"


def test_plain_file_url_is_not_extracted(tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve("{}")):
            path = cached_download(FILE_URL, "llm/thing.json")
    assert path == tmp_path / "custom" / "llm" / "thing.json"
    assert path.read_text() == "{}"
    assert not (path.parent / ".downloaded.ok").exists()


def test_cache_miss_is_visible_by_default(tmp_path, monkeypatch, caplog):
    """The cache-miss notice must be a WARNING, not INFO: it needs to show up
    with no logging configuration at all, the way the old bespoke
    MASTER_MIND_CACHE-based downloader always printed it."""
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        with caplog.at_level("WARNING", logger="cached_hub"):
            with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve()):
                cached_download(URL, "llm/thing")
    assert "llm/thing" in caplog.text and ENV_PATH in caplog.text


def test_interrupted_archive_download_is_not_left_behind(tmp_path):
    """A download that dies partway through (network error, ^C, kill) must not
    leave a file at the expected archive path -- otherwise the next call sees
    `archive.exists()` and reuses the truncated file forever instead of
    retrying, which is exactly how a corrupted .tar.gz used to get stuck in
    the cache."""

    def _dies_after_writing_garbage(url, dest):
        dest.write_bytes(b"not a valid gzip stream")
        raise ConnectionError("connection reset")

    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch(
            "cached_hub.web.urlretrieve", side_effect=_dies_after_writing_garbage
        ):
            with pytest.raises(ConnectionError):
                cached_download(URL, "llm/thing")

    archive = tmp_path / "custom" / "llm" / "thing.tar.gz"
    assert not archive.exists()
    assert list(archive.parent.glob(".thing.tar.gz.*")) == []

    # A retry with a working download now succeeds instead of reusing garbage.
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with patch("cached_hub.web.urlretrieve", side_effect=_fake_urlretrieve()):
            path = cached_download(URL, "llm/thing")
    assert (path / "data.txt").read_text() == "hello"


def test_enforce_mode_raises_instead_of_downloading(tmp_path):
    env = {ENV_PATH: str(tmp_path), ENV_ENFORCE: "1"}
    with patch.dict(os.environ, env, clear=True):
        with patch("cached_hub.web.urlretrieve") as urlretrieve:
            with pytest.raises(CacheMissError):
                cached_download(URL, "llm/thing")
    urlretrieve.assert_not_called()
