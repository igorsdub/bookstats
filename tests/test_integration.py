"""Integration test for end-to-end file processing boundary."""

from pathlib import Path

import polars as pl

from bookstats.counts import process_book_file


def test_process_book_file_creates_expected_csv(tmp_path: Path):
    sample_text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "The fox was quick!\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    )
    input_file = tmp_path / "sample.txt"
    input_file.write_text(sample_text, encoding="utf-8")

    output_file = tmp_path / "counts.csv"
    process_book_file(input_file, output_file)

    assert output_file.exists()

    loaded_df = pl.read_csv(output_file)
    assert "word" in loaded_df.columns
    assert "count" in loaded_df.columns

    counts_dict = dict(zip(loaded_df["word"], loaded_df["count"]))
    assert counts_dict["the"] == 3
    assert counts_dict["fox"] == 2
    assert counts_dict["quick"] == 2
