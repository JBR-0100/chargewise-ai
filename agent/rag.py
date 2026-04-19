"""
agent/rag.py
Retrieval-Augmented Generation — Step 2 of the agentic pipeline.
Lightweight TF-IDF retriever over local EV planning guidelines.
No external vector database required.
"""

from __future__ import annotations

import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Path to knowledge base ─────────────────────────────────────────────
DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge",
    "ev_planning_guidelines.txt",
)


def _load_chunks(kb_path: str) -> list[str]:
    """Split guideline text into section-level chunks."""
    with open(kb_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on SECTION headers
    chunks = re.split(r"\n(?=SECTION \d+:)", raw.strip())
    return [c.strip() for c in chunks if c.strip()]


class EVKnowledgeBase:
    """TF-IDF retriever over EV planning guidelines."""

    def __init__(self, kb_path: str = DEFAULT_KB_PATH):
        self.chunks = _load_chunks(kb_path)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.95,
            min_df=1,
        )
        self._matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Return the top-k most relevant guideline chunks for the query.

        Parameters
        ----------
        query   : free-text query derived from demand stats
        top_k   : number of chunks to return
        """
        q_vec = self.vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0.0]

    def retrieve_for_stats(self, stats) -> list[str]:
        """
        Build a context-aware query from ZoneDemandStats and retrieve.
        """
        parts = [
            f"EV charging demand zone {'high load' if stats.is_high_load else 'normal load'}",
            f"peak hour {stats.peak_hour}",
            f"average hourly demand {stats.avg_hourly_kwh:.1f} kWh",
        ]
        if stats.data_quality in ("sparse", "noisy"):
            parts.append("incomplete noisy sensor data")
        if stats.weekday_avg > stats.weekend_avg * 1.2:
            parts.append("weekday commuter charging workplace")
        if stats.peak_hour >= 18 or stats.peak_hour <= 6:
            parts.append("evening peak fast charger deployment")

        query = " ".join(parts)
        return self.retrieve(query, top_k=3)


# ── Singleton cache (avoids re-fitting on every call) ─────────────────
_kb_instance: EVKnowledgeBase | None = None


def get_knowledge_base(kb_path: str = DEFAULT_KB_PATH) -> EVKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = EVKnowledgeBase(kb_path)
    return _kb_instance


# ── Smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    kb = get_knowledge_base()
    results = kb.retrieve("high load evening peak fast charger capacity expansion", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n── Result {i} ──")
        print(r[:300], "...")
