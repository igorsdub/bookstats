"""Unit tests for Zipf's law fitting functions."""

from pathlib import Path

import polars as pl
import pytest

from bookstats.zipf import compute_zipf_fit, fit_all_books


def test_compute_zipf_fit_linear_decay():
    # Construct synthetic data with perfect 1/rank Zipf decay
    ranks = list(range(1, 11))
    counts = [int(1000 / r) for r in ranks]
    df = pl.DataFrame({"word": [f"w{i}" for i in ranks], "count": counts})

    fit = compute_zipf_fit(df)

    # Slope should be approximately -1.0 with high R^2
    assert pytest.approx(fit.slope, rel=0.1) == -1.0
    assert fit.r_squared > 0.95
    assert "rank" in fit.data.columns
    assert "log_rank" in fit.data.columns
    assert "log_count" in fit.data.columns
    assert "fitted_log_count" in fit.data.columns
    assert "fitted_count" in fit.data.columns


def test_compute_zipf_fit_empty():
    df = pl.DataFrame(
        {"word": [], "count": []}, schema={"word": pl.String, "count": pl.UInt32}
    )
    fit = compute_zipf_fit(df)
    assert fit.slope == 0.0
    assert fit.intercept == 0.0
    assert fit.r_squared == 0.0
    assert len(fit.data) == 0


def test_fit_all_books(tmp_path: Path):
    sample_data = pl.DataFrame(
        {
            "book": ["book1", "book1", "book2", "book2"],
            "word": ["the", "and", "the", "in"],
            "count": [100, 50, 80, 40],
        }
    )
    input_file = tmp_path / "book-counts.csv"
    output_file = tmp_path / "zipf-fits.csv"
    sample_data.write_csv(input_file)

    summary = fit_all_books(input_file, output_file)

    assert output_file.exists()
    assert len(summary) == 2
    assert "book" in summary.columns
    assert "slope" in summary.columns
    assert "r_squared" in summary.columns
    assert "total_words" in summary.columns
    assert "unique_words" in summary.columns
    assert summary["book"].to_list() == ["book1", "book2"]
