import pandas as pd
import pytest

from snowpark_batch.normalize import normalize_name, normalize_series


# All flags enabled — matches the old "full pipeline" behavior
ALL_FLAGS = dict(
    replace_symbols=True,
    transliterate_unicode=True,
    lowercase=True,
    remove_punctuation=True,
    remove_suffixes=True,
)


class TestNormalizeName:
    def test_none_returns_empty(self):
        assert normalize_name(None) == ""

    def test_empty_string(self):
        assert normalize_name("") == ""
        assert normalize_name("   ") == ""

    def test_whitespace_trimming(self):
        # Basic whitespace trimming always applies
        assert normalize_name("  hello world  ") == "hello world"

    def test_collapse_multiple_spaces(self):
        # Collapsing spaces always applies
        assert normalize_name("hello    world") == "hello world"

    def test_no_flags_preserves_text(self):
        # With no flags, only whitespace cleanup happens
        assert normalize_name("  Müller & Söhne, GmbH  ") == "Müller & Söhne, GmbH"

    def test_symbol_ampersand(self):
        assert normalize_name("Tom & Jerry", replace_symbols=True) == "Tom and Jerry"

    def test_symbol_plus(self):
        assert normalize_name("A + B", replace_symbols=True) == "A plus B"

    def test_symbol_at(self):
        assert normalize_name("user @ domain", replace_symbols=True) == "user at domain"

    def test_unicode_german(self):
        assert normalize_name("Müller", transliterate_unicode=True) == "Muller"
        assert normalize_name("Straße", transliterate_unicode=True) == "Strasse"
        assert normalize_name("Über", transliterate_unicode=True) == "Uber"

    def test_unicode_french(self):
        assert normalize_name("Café", transliterate_unicode=True) == "Cafe"
        assert normalize_name("François", transliterate_unicode=True) == "Francois"

    def test_unicode_nordic(self):
        assert normalize_name("Ström", transliterate_unicode=True) == "Strom"
        assert normalize_name("Ångström", transliterate_unicode=True) == "Angstrom"

    def test_unicode_various(self):
        assert normalize_name("À", transliterate_unicode=True) == "A"
        assert normalize_name("Ö", transliterate_unicode=True) == "O"
        assert normalize_name("Ú", transliterate_unicode=True) == "U"
        assert normalize_name("Ü", transliterate_unicode=True) == "U"
        assert normalize_name("ç", transliterate_unicode=True) == "c"

    def test_lowercase(self):
        assert normalize_name("Hello World", lowercase=True) == "hello world"

    def test_suffix_removal_llc(self):
        result = normalize_name("Acme LLC", lowercase=True, remove_suffixes=True)
        assert result == "acme"

    def test_suffix_removal_inc_with_period(self):
        result = normalize_name(
            "Acme, Inc.",
            lowercase=True,
            remove_punctuation=True,
            remove_suffixes=True,
        )
        assert result == "acme"

    def test_suffix_removal_ltd(self):
        result = normalize_name("Global Ltd", lowercase=True, remove_suffixes=True)
        assert result == "global"

    def test_suffix_removal_gmbh(self):
        result = normalize_name("Deutsche GmbH", lowercase=True, remove_suffixes=True)
        assert result == "deutsche"

    def test_suffix_not_removed_from_middle(self):
        # "co" should not be removed from the middle of a word
        result = normalize_name("coco beans", lowercase=True, remove_suffixes=True)
        assert result == "coco beans"

    def test_punctuation_removal(self):
        result = normalize_name(
            "Hello, World! (test)",
            lowercase=True,
            remove_punctuation=True,
        )
        assert result == "hello world test"

    def test_combined_pipeline(self):
        result = normalize_name("  Müller & Söhne, GmbH  ", **ALL_FLAGS)
        assert result == "muller and sohne"


class TestNormalizeSeries:
    def test_series(self):
        s = pd.Series(["Café LLC", "Müller & Co", None, ""])
        result = normalize_series(s, **ALL_FLAGS)
        assert result.iloc[0] == "cafe"
        assert result.iloc[1] == "muller and"
        assert result.iloc[2] == ""
        assert result.iloc[3] == ""

    def test_series_no_flags(self):
        s = pd.Series(["Café LLC", "Müller & Co"])
        result = normalize_series(s)
        assert result.iloc[0] == "Café LLC"
        assert result.iloc[1] == "Müller & Co"
