"""Unit tests for Zipf's law fitting functions."""

from pathlib import Path

import polars as pl
import pytest

from bookstats.zipf import compute_zipf_fit, fit_all_books, generate_zipf_chart


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


def test_compute_zipf_fit_summary_metrics_and_line_data():
    df = pl.DataFrame(
        {
            "word": ["the", "of", "and", "to", "a"],
            "count": [100, 50, 25, 10, 5],
        }
    )
    fit = compute_zipf_fit(df, book_name="test_book")
    assert fit.total_words == 190
    assert fit.unique_words == 5
    assert fit.book == "test_book"
    assert len(fit.line_data) == 2
    assert "log_rank" in fit.line_data.columns
    assert "fitted_log_count" in fit.line_data.columns


def test_generate_zipf_chart_single_fit():
    df = pl.DataFrame(
        {
            "word": ["the", "of", "and"],
            "count": [30, 20, 10],
        }
    )
    fit = compute_zipf_fit(df, book_name="sample")
    chart = generate_zipf_chart(fit)
    assert chart is not None
    # Altair chart has to_dict representation
    spec = chart.to_dict()
    assert "layer" in spec or "mark" in spec


def test_generate_zipf_chart_multi_book():
    sample_data = pl.DataFrame(
        {
            "book": ["book1", "book1", "book2", "book2"],
            "word": ["the", "and", "the", "in"],
            "count": [100, 50, 80, 40],
        }
    )
    chart = generate_zipf_chart(sample_data)
    assert chart is not None
    spec = chart.to_dict()
    assert "layer" in spec or "mark" in spec
