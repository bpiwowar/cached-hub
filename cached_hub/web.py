"""Generic URL / archive caching, for resources that do not fit the
HuggingFace layout in :mod:`cached_hub.config` (course-specific data
published as a plain file or a ``.tar.gz`` on a course web server).

Shares the same ``CACHED_HUB_PATH`` root and the same cache-miss policy
(``CACHED_HUB_ENFORCE``) as the HuggingFace loaders, plus a per-user temp
directory fallback for shared caches that a student can read but not write
to::

    from cached_hub import cached_download

    path = cached_download(
        "https://example.org/data/lotte-technology-100000.tar.gz",
        "llm/lotte-technology-100000",
    )

Layout: ``<CACHED_HUB_PATH>/custom/<name>`` -- a fixed ``custom/`` subfolder,
parallel to the ``huggingface/`` one HF resources use (see
:mod:`cached_hub.config`), so a course choosing a ``name`` can never collide
with the HF layout.
"""

from __future__ import annotations

import getpass
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from .config import (
    ENV_PATH,
    CacheMissError,
    get_cache_path,
    is_download_complete,
    is_enforce_mode,
    mark_download_complete,
)

logger = logging.getLogger("cached_hub")

#: Subfolder generic downloads live under, in a configured cache root --
#: parallel to the "huggingface" one HF resources use, so the two can never
#: collide regardless of what a course names its own resources.
CUSTOM_DIR = "custom"


def _is_archive(url: str) -> bool:
    return url.endswith(".tar.gz") or url.endswith(".tgz")


def _is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _download_to(url: str, dest: Path) -> None:
    """Download ``url`` into a temp file next to ``dest``, then move it into place.

    ``urlretrieve`` writing straight to ``dest`` would leave a truncated file
    there if the download is interrupted (network error, ^C, killed process)
    -- and callers treat ``dest.exists()`` as "already downloaded", so a
    partial file gets silently reused forever afterwards instead of retried.
    Downloading to a sibling temp file and renaming only on success means a
    half-finished download never appears at the name callers check.
    """
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urlretrieve(url, tmp_path)
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _fetch(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest``, extracting a ``.tar.gz``/``.tgz`` archive.

    A local archive already sitting at ``dest.with_suffix(".tar.gz")`` (e.g.
    built by a course's own data-generation script) is reused instead of
    re-downloaded, and is never deleted -- only an archive this function
    itself downloaded is cleaned up afterwards.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_archive(url):
        archive = dest.with_suffix(".tar.gz")
        downloaded = not archive.exists()
        if downloaded:
            _download_to(url, archive)
        else:
            logger.info("Using existing archive %s", archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=dest, filter="data")
        if downloaded:
            archive.unlink()
        mark_download_complete(dest)
    else:
        _download_to(url, dest)
    return dest


def cached_download(url: str, name: str) -> Path:
    """Return a local path for ``name``, downloading ``url`` if necessary.

    ``name`` is a relative path (e.g. ``"llm/lotte-technology-100000"``). A
    ``.tar.gz``/``.tgz`` URL is extracted into ``name``, which is then a
    directory; anything else is saved as a single file.

    Resolution order:

    1. ``<CACHED_HUB_PATH>/custom/<name>`` already there -- used as is.
    2. ``CACHED_HUB_PATH`` set, entry missing, root writable -- downloaded
       there. This is what pre-filling the shared cache for a classroom does.
    3. ``CACHED_HUB_PATH`` set, entry missing, root read-only -- downloaded to
       a per-user temp directory instead. Later calls reuse that copy, but it
       does not survive a reboot -- the usual case on lab machines, where
       students may read the shared cache but not write to it.
    4. ``CACHED_HUB_PATH`` unset -- ``outputs/cache/<name>``, relative to the
       current directory, so pre-downloading and running the notebook from
       different directories downloads the file twice.

    Raises :class:`~cached_hub.config.CacheMissError` in enforce mode
    (``CACHED_HUB_ENFORCE``) instead of downloading.
    """
    root = get_cache_path()

    def complete(path: Path) -> bool:
        return is_download_complete(path) if _is_archive(url) else path.exists()

    def fetch_or_raise(path: Path) -> Path:
        if is_enforce_mode():
            raise CacheMissError(f"'{name}' not found in local cache at {root}")
        return _fetch(url, path)

    if root is None:
        cache_path = Path("outputs/cache") / name
        if complete(cache_path):
            logger.info("Found %s", cache_path)
            return cache_path
        # A cache miss, same as warn_cache_miss's, so it is visible by
        # default (WARNING) rather than needing logging configured: this is
        # exactly the message that flags an unwanted download.
        logger.warning(
            "%s is not set: %s goes to %s",
            ENV_PATH,
            name,
            cache_path.resolve().parent,
        )
        return fetch_or_raise(cache_path)

    cache_path = root / CUSTOM_DIR / name
    if complete(cache_path):
        logger.info("Found %s", cache_path)
        return cache_path

    if _is_writable(cache_path.parent):
        logger.warning("Did not find %s in %s, downloading it there...", name, root)
        return fetch_or_raise(cache_path)

    username = getpass.getuser()
    temp_path = Path(tempfile.gettempdir()) / username / "cached-hub" / name
    if complete(temp_path):
        logger.info(
            "Did not find %s in %s (read-only), using %s", name, root, temp_path
        )
        return temp_path
    logger.warning(
        "Did not find %s in %s (read-only), downloading to %s", name, root, temp_path
    )
    return fetch_or_raise(temp_path)
