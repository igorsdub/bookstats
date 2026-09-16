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
    total_words : int
        Total word count across all occurrences.
    unique_words : int
        Number of unique vocabulary words.
    data : polars.DataFrame
        DataFrame with columns: rank, count, log_rank, log_count,
        fitted_log_count, fitted_count.
    line_data : polars.DataFrame
        DataFrame with exactly 2 rows (minimum and maximum log_rank)
        and fitted_log_count for fast line rendering.
    book : str or None
        Identifier for the book, if specified.
    """

    slope: float
    intercept: float
    r_squared: float
    total_words: int
    unique_words: int
    data: pl.DataFrame
    line_data: pl.DataFrame
    book: str | None = None


def compute_zipf_fit(
    counts_df: pl.DataFrame, book_name: str | None = None
) -> ZipfFitResult:
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
    book_name : str, optional
        Name or identifier of the book.

    Returns
    -------
    ZipfFitResult
        Fit metrics (slope, intercept, r_squared, counts) and transformed DataFrames.
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
        empty_line_df = pl.DataFrame(
            {"log_rank": [], "fitted_log_count": []},
            schema={"log_rank": pl.Float64, "fitted_log_count": pl.Float64},
        )
        return ZipfFitResult(
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            total_words=0,
            unique_words=0,
            data=empty_df,
            line_data=empty_line_df,
            book=book_name,
        )

    # Sort by count descending and assign rank starting at 1
    sorted_df = counts_df.sort("count", descending=True)
    ranks = np.arange(1, len(sorted_df) + 1, dtype=np.float64)
    counts = sorted_df["count"].to_numpy().astype(np.float64)

    total_words = int(sorted_df["count"].sum())
    unique_words = len(sorted_df)

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

    min_log_rank = float(log_ranks[0])
    max_log_rank = float(log_ranks[-1])
    line_data = pl.DataFrame(
        {
            "log_rank": [min_log_rank, max_log_rank],
            "fitted_log_count": [
                float(slope * min_log_rank + intercept),
                float(slope * max_log_rank + intercept),
            ],
        }
    )
    if book_name is not None:
        line_data = line_data.with_columns(pl.lit(book_name).alias("book"))

    return ZipfFitResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        total_words=total_words,
        unique_words=unique_words,
        data=result_df,
        line_data=line_data,
        book=book_name,
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
        fit = compute_zipf_fit(book_df, book_name=book_id)
        records.append(
            {
                "book": book_id,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.r_squared,
                "total_words": fit.total_words,
                "unique_words": fit.unique_words,
            }
        )

    summary_df = pl.DataFrame(records)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(p_out)
    return summary_df


def generate_zipf_chart(
    data: Path | str | pl.DataFrame | ZipfFitResult,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> alt.TopLevelMixin:
    """Generate an Altair log-log rank-frequency chart with linear fits.

    Supports both single-book fits and multi-book comparisons. Uses 2-point
    regression endpoints to optimize chart size and rendering performance.

    Parameters
    ----------
    data : Path, str, polars.DataFrame, or ZipfFitResult
        Input data: a single ZipfFitResult, a processed counts DataFrame,
        or a Path/str to processed combined book counts CSV.
    title : str, optional
        Custom title for the chart. Defaults to standard titles.
    width : int, optional
        Width of chart in pixels (default: 550 for single book, 600 for multi-book).
    height : int, optional
        Height of chart in pixels (default: 380 for single book, 420 for multi-book).

    Returns
    -------
    altair.TopLevelMixin
        Altair chart combining observed word points and fitted lines.
    """
    if isinstance(data, ZipfFitResult):
        fit = data
        if len(fit.data) == 0:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        # Compact plot data: round floats to reduce payload
        points_data = fit.data.select(
            [
                pl.col("log_rank").round(3),
                pl.col("log_count").round(3),
            ]
            + [c for c in ["word", "rank", "count"] if c in fit.data.columns]
        )
        line_data = fit.line_data.select(
            [
                pl.col("log_rank").round(3),
                pl.col("fitted_log_count").round(3),
            ]
        )

        tooltip_cols = [
            c for c in ["word", "rank", "count"] if c in points_data.columns
        ]

        points = (
            alt.Chart(points_data)
            .mark_circle(size=25, opacity=0.5, color="#1f77b4")
            .encode(
                x=alt.X(
                    "log_rank:Q",
                    title="Log(Rank)",
                    axis=alt.Axis(
                        titleFontSize=14,
                        labelFontSize=12,
                        titlePadding=10,
                    ),
                ),
                y=alt.Y(
                    "log_count:Q",
                    title="Log(Frequency)",
                    axis=alt.Axis(
                        titleFontSize=14,
                        labelFontSize=12,
                        titlePadding=10,
                    ),
                ),
                tooltip=tooltip_cols,
            )
        )

        line = (
            alt.Chart(line_data)
            .mark_line(color="#d62728", strokeDash=[5, 5], strokeWidth=2)
            .encode(
                x=alt.X("log_rank:Q"),
                y=alt.Y("fitted_log_count:Q"),
            )
        )

        chart_title = title or (
            f"Descriptive Zipf Fit: {fit.book}" if fit.book else "Descriptive Zipf Fit"
        )
        subtitle = (
            f"Slope: {fit.slope:.3f} | Intercept: {fit.intercept:.3f} | "
            f"R²: {fit.r_squared:.3f}"
        )

        return (
            (points + line)
            .properties(
                title=alt.Title(
                    chart_title,
                    subtitle=subtitle,
                    fontSize=16,
                    subtitleFontSize=13,
                ),
                width=width or 550,
                height=height or 380,
            )
            .interactive()
        )

    # Multi-book or DataFrame / file path
    if isinstance(data, (str, Path)):
        df = pl.read_csv(data)
    else:
        df = data

    if len(df) == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No data"))

    if "book" in df.columns:
        books = sorted(df["book"].unique().to_list())
        points_frames = []
        line_frames = []

        for book_id in books:
            book_df = df.filter(pl.col("book") == book_id)
            fit = compute_zipf_fit(book_df, book_name=book_id)
            if len(fit.data) > 0:
                p_df = fit.data.select(
                    [
                        pl.lit(book_id).alias("book"),
                        pl.col("log_rank").round(3),
                        pl.col("log_count").round(3),
                    ]
                    + [c for c in ["word", "rank", "count"] if c in fit.data.columns]
                )
                l_df = fit.line_data.select(
                    [
                        pl.lit(book_id).alias("book"),
                        pl.col("log_rank").round(3),
                        pl.col("fitted_log_count").round(3),
                    ]
                )
                points_frames.append(p_df)
                line_frames.append(l_df)

        if not points_frames:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        all_points = pl.concat(points_frames)
        all_lines = pl.concat(line_frames)

        tooltip_cols = [
            c for c in ["book", "word", "rank", "count"] if c in all_points.columns
        ]

        points = (
            alt.Chart(all_points)
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
                tooltip=tooltip_cols,
            )
        )

        lines = (
            alt.Chart(all_lines)
            .mark_line(strokeDash=[4, 4], strokeWidth=2)
            .encode(
                x=alt.X("log_rank:Q"),
                y=alt.Y("fitted_log_count:Q"),
                color=alt.Color("book:N"),
            )
        )

        chart_title = title or "Descriptive Zipf's Law Fit (Log Rank vs Log Frequency)"
        return (
            (points + lines)
            .properties(
                title=alt.Title(chart_title, fontSize=16),
                width=width or 600,
                height=height or 420,
            )
            .interactive()
        )

    # DataFrame without 'book' column: compute as single book fit
    single_fit = compute_zipf_fit(df)
    return generate_zipf_chart(single_fit, title=title, width=width, height=height)


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
