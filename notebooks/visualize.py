import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    from pathlib import Path
    import altair as alt
    import marimo as mo
    import polars as pl
    from bookstats.zipf import compute_zipf_fit

    return Path, alt, compute_zipf_fit, mo, pl


@app.cell
def __(mo):
    mo.md(
        r"""
        # Book Word Frequency & Zipf's Law Analysis

        An interactive analysis of word frequency distributions and descriptive
        Zipf fits across Project Gutenberg books.
        """
    )
    return


@app.cell
def __(Path, pl):
    # Load processed book counts
    processed_path = Path("data/processed/book-counts.csv")
    if processed_path.exists():
        counts_df = pl.read_csv(processed_path)
    else:
        # Fallback to intermediate counts if processed is not yet generated
        intermediate_dir = Path("data/intermediate")
        csv_files = list(intermediate_dir.glob("*.csv"))
        if csv_files:
            frames = []
            for p in csv_files:
                d = pl.read_csv(p).with_columns(pl.lit(p.stem).alias("book"))
                frames.append(d.select(["book", "word", "count"]))
            counts_df = pl.concat(frames)
        else:
            counts_df = pl.DataFrame(
                {"book": [], "word": [], "count": []},
                schema={"book": pl.String, "word": pl.String, "count": pl.UInt32},
            )
    return counts_df, processed_path


@app.cell
def __(counts_df, mo):
    books = (
        sorted(counts_df["book"].unique().to_list()) if len(counts_df) > 0 else ["None"]
    )
    book_selector = mo.ui.dropdown(
        options=books,
        value=books[0] if books else "None",
        label="Select a book:",
    )
    book_selector
    return book_selector, books


@app.cell
def __(alt, book_selector, compute_zipf_fit, counts_df, mo, pl):
    mo.stop(book_selector.value == "None", mo.md("No books available."))

    book_counts = counts_df.filter(pl.col("book") == book_selector.value)
    total_words = int(book_counts["count"].sum())
    unique_words = len(book_counts)

    # Compute descriptive Zipf fit using bookstats.zipf
    fit_result = compute_zipf_fit(book_counts)

    stats = mo.hstack(
        [
            mo.stat(label="Total Words", value=f"{total_words:,}"),
            mo.stat(label="Unique Vocabulary", value=f"{unique_words:,}"),
            mo.stat(label="Fit Slope", value=f"{fit_result.slope:.3f}"),
            mo.stat(
                label="R² (Coeff of Determination)", value=f"{fit_result.r_squared:.3f}"
            ),
        ]
    )

    # Zipf plot: Log(Rank) vs Log(Frequency)
    plot_data = fit_result.data
    points = (
        alt.Chart(plot_data)
        .mark_circle(size=20, opacity=0.5, color="#1f77b4")
        .encode(
            x=alt.X("log_rank:Q", title="Log(Rank)"),
            y=alt.Y("log_count:Q", title="Log(Frequency)"),
            tooltip=["word", "rank", "count"],
        )
    )

    line = (
        alt.Chart(plot_data)
        .mark_line(color="#d62728", strokeDash=[5, 5])
        .encode(
            x=alt.X("log_rank:Q"),
            y=alt.Y("fitted_log_count:Q"),
        )
    )

    zipf_chart = (points + line).properties(
        title=f"Descriptive Zipf Fit: {book_selector.value} (Slope: {fit_result.slope:.2f}, R²: {fit_result.r_squared:.2f})",
        width=600,
        height=400,
    )

    note = mo.md(
        r"""
        > **Note on Descriptive Zipf Fit**:
        > This is an ordinary least squares (OLS) linear fit on log-transformed rank and frequency.
        > While widely used as a descriptive summary, OLS on log-transformed data can yield biased estimates
        > and does not prove that word frequencies strictly follow a power law.
        """
    )

    mo.vstack([stats, zipf_chart, note])
    return (
        book_counts,
        fit_result,
        line,
        note,
        plot_data,
        points,
        stats,
        total_words,
        unique_words,
        zipf_chart,
    )


if __name__ == "__main__":
    app.run()
