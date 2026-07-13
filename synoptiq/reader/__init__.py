"""Koine reading assistant — turn the gold GNT/LXX data + published models into a reader.

Two engines share one :class:`~synoptiq.reader.gold.WordAnalysis` shape:

* **Gold mode** (:mod:`synoptiq.reader.gold`) surfaces the Nestle-1904 GNT and Rahlfs
  LXX Text-Fabric editions on disk — per word: lemma, full morphology, English gloss,
  Strong's number — by reference lookup. Exact, instant, no model inference.
* **Neural mode** (:mod:`synoptiq.reader.neural`, optional — needs torch) analyses
  *arbitrary* Koine (papyri, apocrypha, patristics, a pasted sentence) with the
  published Koine-T5 model, glossing predicted lemmas against the gold lexicon.

:mod:`synoptiq.reader.parallels` adds synoptic-parallel lookup via the Aland table.

The neural engine is intentionally *not* imported here so the gold reader and its
tests stay import-light (no torch dependency).
"""

from __future__ import annotations

from synoptiq.reader.gold import GoldReader, ReadResult, WordAnalysis
from synoptiq.reader.index import IndexReader, load_index, save_index, serialize
from synoptiq.reader.morphology import describe_morphology, tidy_pos
from synoptiq.reader.parallels import find_pericope, parallel_ranges, synoptic_parallels
from synoptiq.reader.textfabric import TFDataset, read_tf_feature, slot_count

__all__ = [
    "GoldReader",
    "IndexReader",
    "ReadResult",
    "TFDataset",
    "WordAnalysis",
    "describe_morphology",
    "find_pericope",
    "load_index",
    "parallel_ranges",
    "read_tf_feature",
    "save_index",
    "serialize",
    "slot_count",
    "synoptic_parallels",
    "tidy_pos",
]
