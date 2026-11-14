"""Shape checks for the SciTableQA loader -- no API key needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import pytest


def _has_cache():
    """Check if the data cache exists (downloaded on first use)."""
    cache = os.path.join(os.path.dirname(__file__), "..", "agent", "data", "scitableqa")
    return os.path.isdir(cache)


@pytest.mark.skipif(not _has_cache(), reason="SciTableQA cache not downloaded")
class TestLoaderShape:
    def test_biology_loads(self):
        from scitableqa_loader import load_domain
        trips = load_domain("Biology")
        assert len(trips) > 400
        t = trips[0]
        assert hasattr(t, "question")
        assert hasattr(t, "gold")
        assert hasattr(t, "table_csv")
        assert hasattr(t, "task_family")
        assert hasattr(t, "grounded")
        assert t.task_family in ("Lookup", "Compute")
        assert isinstance(t.grounded, bool)

    def test_compsci_loads(self):
        from scitableqa_loader import load_domain
        trips = load_domain("CompSci")
        assert len(trips) > 500

    def test_families_present(self):
        from scitableqa_loader import load_domain
        trips = load_domain("Biology")
        families = set(t.task_family for t in trips)
        assert families == {"Lookup", "Compute"}

    def test_grounded_rate_biology(self):
        from scitableqa_loader import load_domain
        trips = load_domain("Biology")
        lookup = [t for t in trips if t.task_family == "Lookup"]
        grounded_rate = sum(t.grounded for t in lookup) / len(lookup)
        # Paper reports ~80%
        assert 0.70 < grounded_rate < 0.90, f"Lookup grounded rate {grounded_rate:.1%}"
