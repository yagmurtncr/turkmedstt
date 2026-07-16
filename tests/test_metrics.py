import pytest

from postprocessing.scripts import speech_shared as s


def test_wer_identical_is_zero():
    assert s.word_error_rate("bir iki uc", "bir iki uc") == 0.0


def test_wer_one_substitution():
    assert s.word_error_rate("bir iki uc", "bir iki dort") == pytest.approx(1 / 3)


def test_wer_empty_reference():
    assert s.word_error_rate("", "bir") == 1.0
    assert s.word_error_rate("", "") == 0.0


def test_cer_one_substitution():
    assert s.char_error_rate("abc", "abd") == pytest.approx(1 / 3)
    assert s.char_error_rate("abc", "abc") == 0.0


def test_edit_distance_matches_classic_levenshtein():
    assert s._edit_distance(list("kitten"), list("sitting")) == 3


def test_wer_bucket_thresholds():
    assert s.wer_bucket(0.10) == "near_clean"
    assert s.wer_bucket(0.11) == "minor_errors"
    assert s.wer_bucket(0.30) == "minor_errors"
    assert s.wer_bucket(0.50) == "moderate_errors"
    assert s.wer_bucket(0.90) == "major_errors"
