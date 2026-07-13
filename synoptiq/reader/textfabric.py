"""Dependency-free Text-Fabric reader for the on-disk gold Greek datasets.

The reading assistant surfaces the Nestle-1904 GNT and Rahlfs LXX Text-Fabric
editions (``data/raw/{n1904,lxx}``). Each ships one file per *feature* in the
Text-Fabric ``.tf`` format: a run of ``@``-prefixed header lines, one blank
separator, then a data section addressing **nodes**. Words are the lowest-numbered
nodes (*slots*, ``1..n``); higher nodes are the syntax tree (book/chapter/verse/
clause/group/phrase).

Unlike the minimal reader in :mod:`synoptiq.data.koine_corpus` — which assumes a
fully dense one-value-per-line column and is correct only for the LXX *section*
features — this reader expands the **compact** encoding used by the sparse
morphology features (``case``/``tense``/… omit nodes that lack the feature). A
data line is one of:

* ``value``            — the next node (previous + 1) takes ``value``;
* ``node<TAB>value``   — an explicit node takes ``value``;
* ``n1-n2<TAB>value``  — every node in the inclusive range takes ``value``.

Nodes with no line keep the empty default. Getting this right matters: the naive
reader silently misaligns ``case`` after the first caseless word (a conjunction,
an indeclinable name), which would then mis-parse every following token.
"""

from __future__ import annotations

from pathlib import Path


def read_tf_feature(path: Path | str) -> dict[int, str]:
    """Parse a Text-Fabric ``@node`` feature file into a ``{node: value}`` map.

    Expands the compact encoding (bare value → previous node + 1; ``n<TAB>value``;
    ``n1-n2<TAB>value`` range). Nodes absent from the file, or written with an empty
    value, are simply not present in the returned mapping.

    Args:
        path: Path to a ``.tf`` feature file.

    Returns:
        Mapping from 1-based node number to its (non-empty) string value.

    Example:
        >>> feat = read_tf_feature("data/raw/n1904/tf/1.0.0/case.tf")
        >>> feat[1]
        'nominative'
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("@"):
        i += 1
    if i < len(lines) and lines[i] == "":  # single blank header/data separator
        i += 1

    data: dict[int, str] = {}
    node = 0
    for line in lines[i:]:
        tab = line.find("\t")
        if tab == -1:  # bare value → next node
            node += 1
            if line:
                data[node] = line
            continue
        spec, value = line[:tab], line[tab + 1 :]
        if "-" in spec:  # inclusive node range
            start_s, _, end_s = spec.partition("-")
            start, end = int(start_s), int(end_s)
            if value:
                for k in range(start, end + 1):
                    data[k] = value
            node = end
        else:  # explicit single node
            node = int(spec)
            if value:
                data[node] = value
    return data


def slot_count(tf_dir: Path | str) -> int:
    """Return the number of word slots in a TF dataset, read from ``otype.tf``.

    Slots are the contiguous run of lowest-numbered nodes sharing the type of node 1
    (``word``); the first ``otype`` range is ``1-<n_slots>``.

    Args:
        tf_dir: Directory holding the ``.tf`` feature files.

    Returns:
        The word-slot count ``n`` (slots are nodes ``1..n``).
    """
    otype = read_tf_feature(Path(tf_dir) / "otype.tf")
    slot_type = otype.get(1)
    n = 0
    for node in sorted(otype):
        if otype[node] == slot_type and node == n + 1:
            n = node
        else:
            break
    return n


class TFDataset:
    """Lazy, cached access to a Text-Fabric dataset directory.

    Feature columns are read on first use and memoised. Only the word-slot span
    (``1..n_slots``) is meaningful for per-word lookups; higher nodes belong to the
    syntax tree and are ignored by the reader.

    Args:
        tf_dir: Directory containing ``otype.tf`` and the feature files.

    Raises:
        FileNotFoundError: If ``tf_dir`` has no ``otype.tf``.
    """

    def __init__(self, tf_dir: Path | str) -> None:
        self.dir = Path(tf_dir)
        if not (self.dir / "otype.tf").exists():
            msg = f"Not a Text-Fabric dataset (no otype.tf): {self.dir}"
            raise FileNotFoundError(msg)
        self.n_slots: int = slot_count(self.dir)
        self._cache: dict[str, dict[int, str]] = {}

    def has(self, feature: str) -> bool:
        """Return True if ``<feature>.tf`` exists in the dataset."""
        return (self.dir / f"{feature}.tf").exists()

    def feature(self, name: str) -> dict[int, str]:
        """Return the (cached) ``{node: value}`` map for a feature (empty if absent)."""
        if name not in self._cache:
            path = self.dir / f"{name}.tf"
            self._cache[name] = read_tf_feature(path) if path.exists() else {}
        return self._cache[name]

    def value(self, feature: str, slot: int) -> str:
        """Return the value of ``feature`` at ``slot`` (empty string if unset)."""
        return self.feature(feature).get(slot, "")
