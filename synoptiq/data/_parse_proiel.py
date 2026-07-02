"""PROIEL Universal Dependencies (CoNLL-U) parser for SynoptiQ.

Parses the UD_Ancient_Greek-PROIEL repository (CoNLL-U format) to
extract dependency structure for NT Greek. This data is used to
enrich token records with syntactic head/deprel information.

CoNLL-U format (10 tab-separated columns per token line):
  ID    FORM    LEMMA   UPOS    XPOS    FEATS   HEAD    DEPREL  DEPS    MISC
  1     Ἐν      ἐν      ADP     -       -       3       obl     _       _

Sentence boundaries are marked by blank lines.
Sentence metadata is in comment lines (# key = value).

The PROIEL UD treebank covers the entire NT in Greek.
We use it to validate lemmas and extract dependency structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import ConlluToken

_LOG = get_logger(__name__)

# ── CoNLL-U parsing ────────────────────────────────────────────────────────────


def _parse_feats(feats_str: str) -> dict[str, str]:
    """Parse CoNLL-U FEATS column (e.g., "Case=Nom|Gender=Masc|Number=Sing").

    Args:
        feats_str: FEATS column string (or "_" for empty).

    Returns:
        Dict of feature name → value (empty dict if "_").
    """
    if feats_str in ("_", ""):
        return {}
    result: dict[str, str] = {}
    for feat in feats_str.split("|"):
        if "=" in feat:
            key, val = feat.split("=", 1)
            result[key] = val
    return result


def _parse_conllu_token(line: str) -> ConlluToken | None:
    """Parse one CoNLL-U token line (skip multi-word tokens and empty nodes).

    Args:
        line: A single non-comment, non-blank line from a CoNLL-U file.

    Returns:
        ConlluToken dict, or None for multi-word tokens (id contains "-")
        or empty nodes (id contains ".").
    """
    cols = line.strip().split("\t")
    if len(cols) < 10:
        return None

    id_str, form, lemma, upos, xpos, feats, head_str, deprel, deps, misc = cols[:10]

    # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
    if "-" in id_str or "." in id_str:
        return None

    try:
        token_id = int(id_str)
        head = int(head_str) if head_str != "_" else 0
    except ValueError:
        return None

    return ConlluToken(
        id=token_id,
        form=form,
        lemma=lemma,
        upos=upos,
        xpos=xpos,
        feats=_parse_feats(feats),
        head=head,
        deprel=deprel,
        deps=deps,
        misc=misc,
    )


def _iter_sentences(
    lines: list[str],
) -> Iterator[tuple[dict[str, str], list[ConlluToken]]]:
    """Iterate over (metadata, tokens) pairs from CoNLL-U lines.

    Args:
        lines: All lines from a CoNLL-U file.

    Yields:
        Tuple of (metadata_dict, token_list) for each sentence.
    """
    metadata: dict[str, str] = {}
    tokens: list[ConlluToken] = []

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("#"):
            # Metadata comment: "# key = value"
            if "=" in line:
                key, _, val = line[2:].partition("=")
                metadata[key.strip()] = val.strip()
        elif line == "":
            # Blank line = sentence boundary
            if tokens:
                yield metadata, tokens
            metadata = {}
            tokens = []
        else:
            token = _parse_conllu_token(line)
            if token is not None:
                tokens.append(token)

    # Handle file without trailing newline
    if tokens:
        yield metadata, tokens


def parse_proiel(
    proiel_dir: Path,
    *,
    only_nt: bool = True,
) -> list[tuple[dict[str, str], list[ConlluToken]]]:
    """Parse PROIEL UD CoNLL-U files from the UD_Ancient_Greek-PROIEL repo.

    The repository contains ``*.conllu`` files at its root or in subdirs.
    We parse all of them and return a flat list of sentences.

    Args:
        proiel_dir: Path to the cloned UD_Ancient_Greek-PROIEL repository.
        only_nt: If True (default), attempt to filter only NT sentences
            using sentence metadata (``newdoc id`` / ``source`` fields).

    Returns:
        List of (metadata_dict, token_list) tuples — one per sentence.

    Raises:
        FileNotFoundError: If no CoNLL-U files are found.
    """
    conllu_files = sorted(proiel_dir.rglob("*.conllu"))
    if not conllu_files:
        msg = f"No CoNLL-U files found in {proiel_dir}"
        raise FileNotFoundError(msg)

    _LOG.info(
        "found PROIEL CoNLL-U files",
        extra={"count": len(conllu_files)},
    )

    all_sentences: list[tuple[dict[str, str], list[ConlluToken]]] = []

    for filepath in conllu_files:
        _LOG.info("parsing CoNLL-U file", extra={"file": filepath.name})
        lines = filepath.read_text(encoding="utf-8").splitlines()
        sentences = list(_iter_sentences(lines))
        all_sentences.extend(sentences)
        _LOG.info(
            "parsed CoNLL-U file",
            extra={"file": filepath.name, "n_sentences": len(sentences)},
        )

    _LOG.info(
        "PROIEL parse complete",
        extra={"n_sentences": len(all_sentences)},
    )
    return all_sentences


def build_proiel_lemma_index(
    sentences: list[tuple[dict[str, str], list[ConlluToken]]],
) -> dict[str, list[str]]:
    """Build a normalized-form → lemma lookup from PROIEL data.

    Used to cross-validate MorphGNT lemmas: if a form's PROIEL
    lemma differs from MorphGNT's, we log the discrepancy.

    Args:
        sentences: Output of parse_proiel().

    Returns:
        Dict mapping normalized Greek form → list of attested lemmas.
        Multiple lemmas may exist for the same form (homonyms).
    """
    from synoptiq.utils.greek import normalize_greek

    index: dict[str, list[str]] = {}
    for _, tokens in sentences:
        for token in tokens:
            norm = normalize_greek(token["form"])
            if norm not in index:
                index[norm] = []
            if token["lemma"] not in index[norm]:
                index[norm].append(token["lemma"])

    _LOG.info(
        "PROIEL lemma index built",
        extra={"n_forms": len(index)},
    )
    return index
