from postprocessing.scripts import speech_shared as s


def test_remove_diacritics_turkish():
    assert s.remove_diacritics("çöğüş") == "cogus"


def test_clean_text_collapses_whitespace():
    assert s.clean_text("  a   b  ") == "a b"


def test_tokenize_words_splits_on_whitespace():
    assert s.tokenize_words("bir  iki") == ["bir", "iki"]


def test_normalize_for_match_is_case_insensitive():
    assert s.normalize_for_match("Merhaba DÜNYA") == s.normalize_for_match("merhaba dünya")
