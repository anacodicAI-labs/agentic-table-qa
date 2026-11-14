"""Unit tests for the scorer: every branch of is_correct, including containment."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from scorer import is_correct, _norm, _num, _close


# ── numeric match ──────────────────────────────────────────────────────────

def test_exact_int():
    assert is_correct("48", "48")

def test_int_with_units():
    assert is_correct("48 nucleotides", "48")

def test_gold_with_units():
    assert is_correct("48", "48 nucleotides")

def test_percentage_match():
    assert is_correct("14.5%", "14.47%")

def test_zero_equals_zero_pct():
    assert is_correct("0%", "0")

def test_scientific_notation():
    assert is_correct("3.42E-04", "0.000342")

def test_negative_match():
    assert is_correct("-5.3", "-5.3")

def test_difference_question_unsigned():
    assert is_correct("-3", "3", question="What is the difference between A and B?")

def test_difference_question_same_sign():
    assert is_correct("3", "3", question="difference in counts")

def test_wrong_number():
    assert not is_correct("49", "48")

def test_large_relative_error():
    assert not is_correct("100", "50")


# ── string match ───────────────────────────────────────────────────────────

def test_exact_string():
    assert is_correct("Forward primer", "Forward primer")

def test_case_insensitive():
    assert is_correct("forward primer", "Forward Primer")

def test_punctuation_stripped():
    assert is_correct("Forward primer.", "Forward primer")


# ── containment ────────────────────────────────────────────────────────────

def test_containment_match():
    assert is_correct("The answer is Forward primer for Exon 3", "Forward primer")

def test_containment_does_not_match_reversed():
    # gold "The answer is Forward primer" should NOT match pred "Forward"
    # because _norm(gold) is not in _norm(pred)
    assert not is_correct("Forward", "The answer is Forward primer")


# ── edge cases ─────────────────────────────────────────────────────────────

def test_none_pred():
    assert not is_correct(None, "48")

def test_none_gold():
    assert not is_correct("48", None)

def test_empty_pred():
    assert not is_correct("", "48")

def test_empty_gold_containment_quirk():
    # Empty string is a substring of everything, so this returns True.
    # Not a problem in practice: no gold in the dataset is empty.
    assert is_correct("48", "")


# ── helpers ────────────────────────────────────────────────────────────────

def test_norm():
    assert _norm("Hello, World! 123") == "helloworld123"

def test_num_integer():
    assert _num("48 nucleotides") == 48.0

def test_num_float():
    assert _num("14.47%") == 14.47

def test_num_none():
    assert _num("no number here") is None

def test_close_equal():
    assert _close(1.0, 1.0)

def test_close_within_tolerance():
    assert _close(14.47, 14.5, rel=0.01)

def test_close_outside_tolerance():
    assert not _close(14.47, 15.0, rel=0.01)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
