"""Descriptive Zipf's law analysis and log-log linear fitting."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from scipy.stats import linregress

alt.data_transformers.disable_max_rows()


@dataclass(frozen=True)
class ZipfFitResult:
    """Results from a descriptive log-log linear fit to word frequencies.

    Attributes
    ----------
    slope : float
        Slope of the descriptive log-log linear fit.
    intercept : float
        Intercept of the descriptive log-log linear fit.
    r_squared : float
        Coefficient of determination (R^2).
    data : polars.DataFrame
        DataFrame with columns: rank, count, log_rank, log_count,
        fitted_log_count, fitted_count.
    """

    slope: float
    intercept: float
    r_squared: float
    data: pl.DataFrame


def compute_zipf_fit(counts_df: pl.DataFrame) -> ZipfFitResult:
    """Compute ranks, log transforms, and descriptive log-log linear fit.

    This performs an ordinary least squares (OLS) linear fit on
    log(frequency) versus log(rank).

    Note: Ordinary least squares on log-transformed power-law data can give
    biased estimates and does not establish that the book follows Zipf's
    law. It is used here as a descriptive summary.

    Parameters
    ----------
    counts_df : polars.DataFrame
        DataFrame containing at least 'count' (and optionally 'word').

    Returns
    -------
    ZipfFitResult
        Fit metrics (slope, intercept, r_squared) and transformed DataFrame.
    """
    if len(counts_df) == 0:
        empty_df = pl.DataFrame(
            {
                "rank": [],
                "count": [],
                "log_rank": [],
                "log_count": [],
                "fitted_log_count": [],
                "fitted_count": [],
            },
            schema={
                "rank": pl.UInt32,
                "count": pl.UInt32,
                "log_rank": pl.Float64,
                "log_count": pl.Float64,
                "fitted_log_count": pl.Float64,
                "fitted_count": pl.Float64,
            },
        )
        return ZipfFitResult(slope=0.0, intercept=0.0, r_squared=0.0, data=empty_df)

    # Sort by count descending and assign rank starting at 1
    sorted_df = counts_df.sort("count", descending=True)
    ranks = np.arange(1, len(sorted_df) + 1, dtype=np.float64)
    counts = sorted_df["count"].to_numpy().astype(np.float64)

    log_ranks = np.log(ranks)
    log_counts = np.log(counts)

    if len(ranks) > 1:
        res = linregress(log_ranks, log_counts)
        slope = float(res.slope)
        intercept = float(res.intercept)
        r_squared = float(res.rvalue**2)
    else:
        slope = 0.0
        intercept = float(log_counts[0]) if len(log_counts) > 0 else 0.0
        r_squared = 1.0

    fitted_log_counts = slope * log_ranks + intercept
    fitted_counts = np.exp(fitted_log_counts)

    result_df = sorted_df.with_columns(
        [
            pl.Series("rank", ranks.astype(np.uint32)),
            pl.Series("log_rank", log_ranks),
            pl.Series("log_count", log_counts),
            pl.Series("fitted_log_count", fitted_log_counts),
            pl.Series("fitted_count", fitted_counts),
        ]
    )

    return ZipfFitResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        data=result_df,
    )


def fit_all_books(
    processed_counts_path: Path | str, output_summary_path: Path | str
) -> pl.DataFrame:
    """Calculate Zipf fits for all books in processed dataset and save summary CSV.

    Parameters
    ----------
    processed_counts_path : Path or str
        Path to processed combined book counts CSV.
    output_summary_path : Path or str
        Path to write output summary CSV.

    Returns
    -------
    polars.DataFrame
        Summary DataFrame with columns: book, slope, intercept, r_squared,
        total_words, unique_words.
    """
    p_in = Path(processed_counts_path)
    p_out = Path(output_summary_path)

    df = pl.read_csv(p_in)
    records = []

    for book_id in sorted(df["book"].unique().to_list()):
        book_df = df.filter(pl.col("book") == book_id)
        total_words = int(book_df["count"].sum())
        unique_words = len(book_df)
        fit = compute_zipf_fit(book_df)
        records.append(
            {
                "book": book_id,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.r_squared,
                "total_words": total_words,
                "unique_words": unique_words,
            }
        )

    summary_df = pl.DataFrame(records)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(p_out)
    return summary_df


def generate_zipf_chart(processed_counts_path: Path | str) -> alt.TopLevelMixin:
    """Generate an Altair log-log rank-frequency chart with linear fits.

    Parameters
    ----------
    processed_counts_path : Path or str
        Path to processed combined book counts CSV.

    Returns
    -------
    altair.TopLevelMixin
        Altair chart combining observed word points and fitted lines.
    """
    df = pl.read_csv(processed_counts_path)
    book_frames = []

    for book_id in sorted(df["book"].unique().to_list()):
        book_df = df.filter(pl.col("book") == book_id)
        fit = compute_zipf_fit(book_df)
        fit_df = fit.data.with_columns(
            [
                pl.lit(book_id).alias("book"),
                pl.lit(fit.slope).alias("slope"),
                pl.lit(fit.r_squared).alias("r_squared"),
            ]
        )
        book_frames.append(fit_df)

    if not book_frames:
        return alt.Chart().mark_text().encode(text=alt.value("No data"))

    all_data = pl.concat(book_frames)

    # Observed points
    points = (
        alt.Chart(all_data)
        .mark_circle(size=20, opacity=0.4)
        .encode(
            x=alt.X(
                "log_rank:Q",
                title="Log(Rank)",
                axis=alt.Axis(titleFontSize=14, labelFontSize=12, titlePadding=10),
            ),
            y=alt.Y(
                "log_count:Q",
                title="Log(Word Count)",
                axis=alt.Axis(titleFontSize=14, labelFontSize=12, titlePadding=10),
            ),
            color=alt.Color("book:N", title="Book"),
            tooltip=["book", "word", "rank", "count"],
        )
    )

    # Fitted lines
    lines = (
        alt.Chart(all_data)
        .mark_line(strokeDash=[4, 4], strokeWidth=2)
        .encode(
            x=alt.X("log_rank:Q"),
            y=alt.Y("fitted_log_count:Q"),
            color=alt.Color("book:N"),
        )
    )

    chart = (
        (points + lines)
        .properties(
            title=alt.Title(
                "Descriptive Zipf's Law Fit (Log Rank vs Log Frequency)",
                fontSize=16,
            ),
            width=600,
            height=420,
        )
        .interactive()
    )
    return chart


def plot_zipf(
    processed_counts_path: Path | str,
    output_svg_path: Path | str,
    output_html_path: Path | str,
) -> None:
    """Generate and save both SVG and HTML representations of the Zipf chart.

    Parameters
    ----------
    processed_counts_path : Path or str
        Path to processed combined book counts CSV.
    output_svg_path : Path or str
        Destination path for SVG figure.
    output_html_path : Path or str
        Destination path for interactive HTML figure.
    """
    chart = generate_zipf_chart(processed_counts_path)
    p_svg = Path(output_svg_path)
    p_html = Path(output_html_path)

    p_svg.parent.mkdir(parents=True, exist_ok=True)
    p_html.parent.mkdir(parents=True, exist_ok=True)

    chart.save(str(p_svg))
    chart.save(str(p_html))


def main() -> None:
    """Command-line interface for Zipf analysis."""
    parser = argparse.ArgumentParser(
        description="Compute descriptive Zipf fits and generate figures."
    )
    parser.add_argument(
        "input",
        help="Input processed book-counts.csv file.",
    )
    parser.add_argument(
        "--table",
        help="Output summary table CSV file (e.g. output/tables/zipf-fits.csv).",
    )
    parser.add_argument(
        "--svg",
        help="Output static SVG figure (e.g. output/figures/zipf-law.svg).",
    )
    parser.add_argument(
        "--html",
        help="Output interactive HTML figure (e.g. output/figures/zipf-law.html).",
    )

    args = parser.parse_args()

    if args.table:
        fit_all_books(args.input, args.table)
        print(f"Wrote Zipf fit summary table to {args.table}")

    if args.svg and args.html:
        plot_zipf(args.input, args.svg, args.html)
        print(f"Saved figures to {args.svg} and {args.html}")
    elif args.svg or args.html:
        chart = generate_zipf_chart(args.input)
        if args.svg:
            chart.save(args.svg)
            print(f"Saved SVG figure to {args.svg}")
        if args.html:
            chart.save(args.html)
            print(f"Saved HTML figure to {args.html}")

    if not args.table and not args.svg and not args.html:
        # Default behavior if single output arg is given as table
        print(
            "Please specify --table, --svg, or --html.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
