"""Study design: full-triple folds, overlap partition, census, preregistration freeze.

The source-criticism study has two tracks (see ``docs/SOURCE_CRITICISM_STUDY.md``):

* **Track A** — Q reconstruction: train a Fusion-in-Decoder to rebuild Mark from
  Matthew+Luke where ground truth exists (triple tradition), then apply it to the
  double tradition to emit a candidate proto-Q.
* **Track B** — source identification: compare generative models of the double
  tradition whose redactor operators are learned only on triple-tradition
  supervision that *every* live hypothesis accepts (Mk→Mt, Mk→Lk).

Both tracks need the same deterministic scaffolding, defined here and nowhere
else so it can be frozen and hashed:

* the set of **full triples** (pericopes with Matthew, Mark AND Luke all present)
  — the only units on which the redactor operators can be trained and E2 scored;
* deterministic, genre-stratified **k-fold** assignment over those units;
* the **double-tradition freeze list** — scored exactly once (Track B, E1);
* the **Mark-Q overlap partition** for the E2 difference-in-differences test;
* per-unit token weights (used as heteroscedastic weights in the power analysis).

Nothing here imports torch: it is pure corpus bookkeeping, cheap to test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from synoptiq.utils.constants import (
    MARK_Q_OVERLAP_CORE,
    MARK_Q_OVERLAP_EXTENDED,
)
from synoptiq.utils.logging_ import get_logger

if TYPE_CHECKING:
    from synoptiq.data.corpus import Corpus

_LOG = get_logger(__name__)

_TRIPLE_BOOKS: tuple[str, str, str] = ("Matthew", "Mark", "Luke")


# ── Unit records ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TripleUnit:
    """One full-triple pericope: the atomic unit of triple-tradition evaluation.

    ``token_counts`` excludes punctuation and is used both to size training
    batches and as the inverse-variance weight in the power analysis (a longer
    pericope yields a lower-variance per-pericope statistic).
    """

    pericope_id: str
    genre: str
    token_counts: dict[str, int]  # book → non-punctuation token count
    is_overlap_core: bool         # Mark-Q overlap (undisputed)
    is_overlap_extended: bool     # Mark-Q overlap (core ∪ debated)

    @property
    def min_tokens(self) -> int:
        """Smallest per-book token count — the binding size for alignment/scoring."""
        return min(self.token_counts[b] for b in _TRIPLE_BOOKS)


@dataclass(frozen=True)
class DoubleUnit:
    """One double-tradition pericope (Matthew + Luke, no Markan parallel)."""

    pericope_id: str
    genre: str
    token_counts: dict[str, int]  # Matthew / Luke


# ── Membership ──────────────────────────────────────────────────────────────


def _token_counts(corpus: Corpus, pericope_id: str) -> dict[str, int]:
    """Non-punctuation token count per synoptic book for one pericope."""
    counts: dict[str, int] = {}
    for book in _TRIPLE_BOOKS:
        toks = corpus.get_tokens(book, pericope_id=pericope_id, exclude_punctuation=True)  # type: ignore[arg-type]
        counts[book] = len(toks)
    return counts


def full_triples(corpus: Corpus) -> list[TripleUnit]:
    """Return triple-tradition pericopes with Matthew, Mark AND Luke all present.

    A pericope classified ``triple`` can still be missing a book's tokens (the
    parallel exists in the Aland table but did not survive alignment/parsing).
    Those partial units cannot train a three-way operator and are excluded from
    every Phase-5 count — this is why the effective N is smaller than the raw
    triple-tradition pericope count.
    """
    units: list[TripleUnit] = []
    for pericope in corpus.iter_pericopes(tradition="triple"):
        pid = pericope["pericope_id"]
        counts = _token_counts(corpus, pid)
        if all(counts[b] > 0 for b in _TRIPLE_BOOKS):
            units.append(
                TripleUnit(
                    pericope_id=pid,
                    genre=pericope["genre"],
                    token_counts=counts,
                    is_overlap_core=pid in MARK_Q_OVERLAP_CORE,
                    is_overlap_extended=pid in MARK_Q_OVERLAP_CORE + MARK_Q_OVERLAP_EXTENDED,
                )
            )
    units.sort(key=lambda u: u.pericope_id)
    return units


def double_tradition_units(corpus: Corpus) -> list[DoubleUnit]:
    """Return double-tradition pericopes with both Matthew and Luke present.

    This is the **frozen evaluation set** for Track B / E1 — scored exactly once.
    """
    units: list[DoubleUnit] = []
    for pericope in corpus.iter_pericopes(tradition="double"):
        pid = pericope["pericope_id"]
        counts = {
            "Matthew": len(
                corpus.get_tokens("Matthew", pericope_id=pid, exclude_punctuation=True)
            ),
            "Luke": len(corpus.get_tokens("Luke", pericope_id=pid, exclude_punctuation=True)),
        }
        if counts["Matthew"] > 0 and counts["Luke"] > 0:
            units.append(DoubleUnit(pericope_id=pid, genre=pericope["genre"], token_counts=counts))
    units.sort(key=lambda u: u.pericope_id)
    return units


# ── Folds ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoldPlan:
    """Deterministic k-fold assignment over full-triple pericopes.

    ``assignment`` maps pericope_id → fold index in ``[0, n_folds)``. The plan is
    genre-stratified: each genre's units are distributed round-robin across folds
    after a seeded shuffle, so no fold lacks a genre. Overlap pericopes
    are additionally balanced so the E2 DiD test has overlap units in every fold.
    """

    n_folds: int
    seed: int
    assignment: dict[str, int]

    def train_ids(self, held_out: int) -> list[str]:
        """Pericope ids in every fold except ``held_out``."""
        return sorted(p for p, f in self.assignment.items() if f != held_out)

    def test_ids(self, held_out: int) -> list[str]:
        """Pericope ids in the held-out fold."""
        return sorted(p for p, f in self.assignment.items() if f == held_out)


def _stable_shuffle(items: list[str], seed: int, salt: str) -> list[str]:
    """Deterministic shuffle keyed by a blake2b hash — reproducible across runs.

    Avoids ``random`` global state and is stable regardless of Python hash seed.
    """
    def key(x: str) -> str:
        return hashlib.blake2b(f"{seed}:{salt}:{x}".encode(), digest_size=16).hexdigest()

    return sorted(items, key=key)


def build_folds(units: list[TripleUnit], *, n_folds: int, seed: int) -> FoldPlan:
    """Assign full-triple units to ``n_folds`` genre- and overlap-balanced folds.

    Stratification strategy: shuffle within each (overlap-status, genre) stratum
    deterministically, then deal round-robin into folds. This guarantees (a) no
    fold missing a genre it could contain, and (b) Mark-Q overlap pericopes
    spread across folds so every E2 fold can compute the DiD contrast.
    """
    if n_folds < 2:
        msg = f"n_folds must be >= 2, got {n_folds}"
        raise ValueError(msg)
    if len(units) < n_folds:
        msg = f"cannot make {n_folds} folds from {len(units)} units"
        raise ValueError(msg)

    strata: dict[tuple[bool, str], list[str]] = defaultdict(list)
    for u in units:
        strata[(u.is_overlap_core, u.genre)].append(u.pericope_id)

    assignment: dict[str, int] = {}
    # Deal each stratum round-robin, staggering the starting fold per stratum so
    # small strata don't all pile into fold 0.
    for offset, (stratum, ids) in enumerate(sorted(strata.items())):
        shuffled = _stable_shuffle(ids, seed, salt=f"{stratum[0]}|{stratum[1]}")
        for i, pid in enumerate(shuffled):
            assignment[pid] = (i + offset) % n_folds

    return FoldPlan(n_folds=n_folds, seed=seed, assignment=assignment)


# ── Overlap partition (E2 difference-in-differences) ──────────────────────────


def overlap_partition(
    units: list[TripleUnit], *, scope: str = "core"
) -> tuple[list[str], list[str]]:
    """Split full triples into (Mark-Q overlap, rest) for the E2 DiD contrast.

    Args:
        units: Full-triple units.
        scope: ``"core"`` (undisputed overlaps only) or ``"extended"``.

    Returns:
        (overlap_ids, rest_ids), each sorted.
    """
    if scope not in {"core", "extended"}:
        msg = f"scope must be 'core' or 'extended', got {scope!r}"
        raise ValueError(msg)
    pick = (lambda u: u.is_overlap_core) if scope == "core" else (lambda u: u.is_overlap_extended)
    overlap = sorted(u.pericope_id for u in units if pick(u))
    rest = sorted(u.pericope_id for u in units if not pick(u))
    return overlap, rest


# ── Census ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Census:
    """Every count Phase 5 depends on, computed from the corpus in one pass."""

    n_full_triples: int
    n_double: int
    triple_token_totals: dict[str, int]
    double_token_totals: dict[str, int]
    triple_genres: dict[str, int]
    double_genres: dict[str, int]
    n_overlap_core: int
    n_overlap_extended: int
    # Genres present in the double tradition but absent from the triple tradition
    # (unlearnable a priori — no supervision for the operator on that stratum).
    unlearnable_double_genres: list[str]
    triple_min_tokens: dict[str, int]  # pericope_id → binding (smallest-book) size

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view for reports and hashing."""
        return {
            "n_full_triples": self.n_full_triples,
            "n_double": self.n_double,
            "triple_token_totals": self.triple_token_totals,
            "double_token_totals": self.double_token_totals,
            "triple_genres": self.triple_genres,
            "double_genres": self.double_genres,
            "n_overlap_core": self.n_overlap_core,
            "n_overlap_extended": self.n_overlap_extended,
            "unlearnable_double_genres": self.unlearnable_double_genres,
        }


def census(corpus: Corpus) -> Census:
    """Compute the Phase-5 census from a loaded corpus."""
    triples = full_triples(corpus)
    doubles = double_tradition_units(corpus)

    triple_genres = dict(Counter(u.genre for u in triples))
    double_genres = dict(Counter(u.genre for u in doubles))
    unlearnable = sorted(set(double_genres) - set(triple_genres))

    triple_totals = {
        b: int(sum(u.token_counts[b] for u in triples)) for b in _TRIPLE_BOOKS
    }
    double_totals = {
        b: int(sum(u.token_counts[b] for u in doubles)) for b in ("Matthew", "Luke")
    }

    return Census(
        n_full_triples=len(triples),
        n_double=len(doubles),
        triple_token_totals=triple_totals,
        double_token_totals=double_totals,
        triple_genres=triple_genres,
        double_genres=double_genres,
        n_overlap_core=sum(u.is_overlap_core for u in triples),
        n_overlap_extended=sum(u.is_overlap_extended for u in triples),
        unlearnable_double_genres=unlearnable,
        triple_min_tokens={u.pericope_id: u.min_tokens for u in triples},
    )


# ── Freezing / hashing (preregistration) ──────────────────────────────────────


def _canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace jitter."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(obj: object) -> str:
    """SHA-256 of the canonical JSON of ``obj`` — the preregistration primitive."""
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def config_hash(config_asdict: dict[str, object]) -> str:
    """Hash a ``StudyConfig`` (as a plain dict) for the §10 freeze block.

    ``Path`` values must be stringified by the caller (``dataclasses.asdict``
    leaves them as ``Path``); we coerce here for safety.
    """
    coerced = {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in config_asdict.items()}
    return sha256(coerced)


def fold_hash(plan: FoldPlan) -> str:
    """Hash a fold plan so a re-derived plan can be verified identical to the frozen one."""
    return sha256({"n_folds": plan.n_folds, "seed": plan.seed, "assignment": plan.assignment})


def overlap_hash(scope: str = "core") -> str:
    """Hash the frozen Mark-Q overlap id list."""
    ids = MARK_Q_OVERLAP_CORE if scope == "core" else MARK_Q_OVERLAP_CORE + MARK_Q_OVERLAP_EXTENDED
    return sha256(sorted(ids))
