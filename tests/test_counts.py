"""Unit tests for word extraction and counting functions."""

from bookstats.counts import count_words, extract_words


def test_extract_words_normalizes_text():
    assert extract_words("Hello, HELLO! World?") == [
        "hello",
        "hello",
        "world",
    ]


def test_extract_words_empty_string():
    assert extract_words("") == []


def test_count_words():
    words = ["apple", "banana", "apple", "cherry", "apple", "banana"]
    df = count_words(words)
    assert df["word"].to_list() == ["apple", "banana", "cherry"]
    assert df["count"].to_list() == [3, 2, 1]


def test_count_words_empty():
    df = count_words([])
    assert len(df) == 0
    assert "word" in df.columns
    assert "count" in df.columns
