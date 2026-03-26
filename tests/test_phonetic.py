import pandas as pd

from snowpark_batch.phonetic import phonetic_code, phonetic_series


class TestPhoneticCode:
    def test_empty_string(self):
        assert phonetic_code("") == ""

    def test_basic_name(self):
        code = phonetic_code("smith")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_similar_sounding_names_match(self):
        # "smith" and "smyth" should produce the same phonetic code
        assert phonetic_code("smith") == phonetic_code("smyth")

    def test_different_names_differ(self):
        assert phonetic_code("smith") != phonetic_code("jones")

    def test_known_metaphone_output(self):
        # Metaphone for "john" is "JN"
        assert phonetic_code("john") == "JN"

    def test_phonetic_grouping(self):
        # Names starting with similar sounds should share prefixes
        c1 = phonetic_code("johnson")
        c2 = phonetic_code("jonson")
        assert c1 == c2


class TestPhoneticSeries:
    def test_series(self):
        s = pd.Series(["smith", "jones", ""])
        result = phonetic_series(s)
        assert result.iloc[0] == phonetic_code("smith")
        assert result.iloc[1] == phonetic_code("jones")
        assert result.iloc[2] == ""
