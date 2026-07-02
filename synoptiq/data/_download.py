"""Corpus downloader for SynoptiQ Phase 1.

Downloads all required corpora via ``git clone --depth 1`` to avoid
pulling full Git history (which can be hundreds of MB for some repos).

If a target directory already exists and is a valid Git repo,
pulls the latest changes instead of cloning from scratch.

Corpora downloaded:
  - SBLGNT: Faithlife/SBLGNT (XML, CC-BY 4.0)
  - MorphGNT: morphgnt/sblgnt (TSV, CC-BY-SA 4.0)
  - PROIEL UD: UniversalDependencies/UD_Ancient_Greek-PROIEL (CoNLL-U, CC-BY-NC-SA)
  - N1904-TF: CenterBLC/N1904 (Text-Fabric, CC-BY-NC-ND)
  - Apostolic Fathers: jtauber/apostolic-fathers (plain text, MIT)
  - First1KGreek: OpenGreekAndLatin/First1KGreek (TEI XML, CC-BY 4.0)
  - LXX: biblicalhumanities/Septuagint (plain text, CC-BY-SA)
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# ── Corpus registry ────────────────────────────────────────────────────────────

CORPORA: dict[str, dict[str, str]] = {
    "sblgnt": {
        "url": "https://github.com/Faithlife/SBLGNT",
        "description": "SBL Greek New Testament (XML, Faithlife)",
        "license": "CC-BY 4.0",
    },
    "morphgnt": {
        "url": "https://github.com/morphgnt/sblgnt",
        "description": "MorphGNT morphological annotations (TSV)",
        "license": "CC-BY-SA 4.0",
    },
    "proiel": {
        "url": "https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL",
        "description": "PROIEL Ancient Greek NT treebank (CoNLL-U/UD)",
        "license": "CC-BY-NC-SA 4.0",
    },
    "n1904": {
        "url": "https://github.com/CenterBLC/N1904",
        "description": "Nestle 1904 GNT in Text-Fabric format (CenterBLC)",
        "license": "CC-BY-NC-ND",
    },
    "apostolic": {
        "url": "https://github.com/jtauber/apostolic-fathers",
        "description": "Open Apostolic Fathers plain text (Tauber/Macdonald)",
        "license": "MIT",
    },
    "first1k": {
        "url": "https://github.com/OpenGreekAndLatin/First1KGreek",
        "description": "First 1000 Years of Greek (TEI XML, incl. Josephus)",
        "license": "CC-BY 4.0",
    },
    "lxx": {
        "url": "https://github.com/biblicalhumanities/Septuagint",
        "description": "Septuagint (Rahlfs LXX) plain text for Koine DAPT",
        "license": "CC-BY-SA",
    },
}


# ── Core download logic ────────────────────────────────────────────────────────


def _is_git_repo(path: Path) -> bool:
    """Return True if ``path`` is the root of a Git repository."""
    return (path / ".git").is_dir()


def _git_clone(url: str, target: Path, *, shallow: bool = True) -> None:
    """Clone ``url`` into ``target``.

    Args:
        url: Git remote URL.
        target: Destination directory.
        shallow: If True, use ``--depth 1`` (single commit, no history).
    """
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [url, str(target)]

    _LOG.info("cloning corpus", extra={"url": url, "target": str(target)})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = f"git clone failed for {url}:\n{result.stderr}"
        raise RuntimeError(msg)
    _LOG.info("clone complete", extra={"target": str(target)})


def _git_pull(repo_dir: Path) -> None:
    """Pull latest changes in an existing Git repository.

    Args:
        repo_dir: Path to the existing local Git repository.
    """
    cmd = ["git", "-C", str(repo_dir), "pull", "--depth", "1", "--ff-only"]
    _LOG.info("pulling latest changes", extra={"repo": str(repo_dir)})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _LOG.warning(
            "git pull failed (using existing clone)",
            extra={"repo": str(repo_dir), "stderr": result.stderr},
        )
    else:
        _LOG.info("pull complete", extra={"repo": str(repo_dir)})


def download_corpus(
    name: str,
    raw_dir: Path,
    *,
    force_reclone: bool = False,
) -> Path:
    """Download a single corpus by name.

    If the corpus directory already exists and is a valid git repository,
    pulls latest changes instead of re-cloning (unless ``force_reclone=True``).

    Args:
        name: Corpus name (must be a key in ``CORPORA``).
        raw_dir: Parent directory for raw data (e.g., ``data/raw/``).
        force_reclone: If True, delete existing directory and re-clone.

    Returns:
        Path to the downloaded corpus directory.

    Raises:
        KeyError: If ``name`` is not a registered corpus.
        RuntimeError: If the download fails.
    """
    if name not in CORPORA:
        msg = f"Unknown corpus: {name!r}. Available: {list(CORPORA)}"
        raise KeyError(msg)

    info = CORPORA[name]
    target = raw_dir / name

    if force_reclone and target.exists():
        import shutil

        _LOG.info("removing existing corpus for re-clone", extra={"target": str(target)})
        shutil.rmtree(target)

    if target.exists() and _is_git_repo(target):
        _LOG.info("corpus already downloaded — pulling latest", extra={"name": name})
        _git_pull(target)
    else:
        target.mkdir(parents=True, exist_ok=True)
        # Remove target (may be empty dir) and re-clone
        import shutil

        shutil.rmtree(target)
        _git_clone(info["url"], target)

    _LOG.info(
        "corpus ready",
        extra={"name": name, "description": info["description"], "path": str(target)},
    )
    return target


def download_all(
    raw_dir: Path,
    *,
    corpora: list[str] | None = None,
    force_reclone: bool = False,
) -> dict[str, Path]:
    """Download all registered corpora.

    Args:
        raw_dir: Parent directory for raw data.
        corpora: Specific corpus names to download. If None, downloads all.
        force_reclone: If True, re-clone even if directory exists.

    Returns:
        Dict mapping corpus name → local path.
    """
    to_download = corpora if corpora is not None else list(CORPORA)
    _LOG.info("starting corpus downloads", extra={"corpora": to_download})

    paths: dict[str, Path] = {}
    for name in to_download:
        try:
            paths[name] = download_corpus(name, raw_dir, force_reclone=force_reclone)
        except Exception as e:
            _LOG.error(
                "corpus download failed",
                extra={"name": name, "error": str(e)},
            )
            raise

    _LOG.info(
        "all corpora downloaded",
        extra={"n": len(paths), "paths": {k: str(v) for k, v in paths.items()}},
    )
    return paths
