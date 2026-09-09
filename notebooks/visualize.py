import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import altair as alt
    import marimo as mo
    import polars as pl
    from bookstats.zipf import compute_zipf_fit

    alt.data_transformers.disable_max_rows()

    return Path, alt, compute_zipf_fit, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Book Word Frequency & Zipf's Law Analysis

    An interactive analysis of word frequency distributions and descriptive
    Zipf fits across Project Gutenberg books.

    ### The Zipf's Law Model

    Zipf's law states that the frequency $f$ of a word is inversely proportional to its rank $r$ in the frequency table:

    $$f(r) \propto \frac{1}{r^s} \quad \text{or} \quad f(r) = \frac{C}{r^s}$$

    Taking the natural logarithm of both sides yields a linear relationship:

    $$\ln(f) = \ln(C) - s \cdot \ln(r)$$

    In this analysis, we perform an ordinary least squares (OLS) linear regression of $\ln(\text{count})$ against $\ln(\text{rank})$:

    $$\ln(\text{count}) = \beta_0 + \beta_1 \ln(\text{rank}) + \epsilon$$

    where the slope $\beta_1 \approx -s$ describes how rapidly word frequency decreases with rank, and $R^2$ indicates how well the linear model fits the log-transformed data.
    """)
    return


@app.cell
def _(Path, pl):
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
    return (counts_df,)


@app.cell
def _(counts_df, mo):
    books = (
        sorted(counts_df["book"].unique().to_list()) if len(counts_df) > 0 else ["None"]
    )
    book_selector = mo.ui.dropdown(
        options=books,
        value=books[0] if books else "None",
        label="Select a book:",
    )
    book_selector
    return (book_selector,)


@app.cell
def _(alt, book_selector, compute_zipf_fit, counts_df, mo, pl):
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
            mo.stat(label="Fit Slope (β₁)", value=f"{fit_result.slope:.3f}"),
            mo.stat(label="R² (Determination)", value=f"{fit_result.r_squared:.3f}"),
        ]
    )

    # Zipf plot: Log(Rank) vs Log(Frequency)
    plot_data = fit_result.data
    points = (
        alt.Chart(plot_data)
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
            tooltip=["word", "rank", "count"],
        )
    )

    line = (
        alt.Chart(plot_data)
        .mark_line(color="#d62728", strokeDash=[5, 5], strokeWidth=2)
        .encode(
            x=alt.X("log_rank:Q"),
            y=alt.Y("fitted_log_count:Q"),
        )
    )

    zipf_chart = (
        (points + line)
        .properties(
            title=alt.Title(
                f"Descriptive Zipf Fit: {book_selector.value}",
                subtitle=(
                    f"Slope: {fit_result.slope:.3f} | Intercept: {fit_result.intercept:.3f} | R²: {fit_result.r_squared:.3f}"
                ),
                fontSize=16,
                subtitleFontSize=13,
            ),
            width=550,
            height=380,
        )
        .interactive()
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
    return


if __name__ == "__main__":
    app.run()
