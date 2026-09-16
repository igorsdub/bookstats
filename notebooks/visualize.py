import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import polars as pl
    from bookstats.zipf import compute_zipf_fit, generate_zipf_chart

    return Path, compute_zipf_fit, generate_zipf_chart, mo, pl


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
def _(Path, mo, pl):
    # Load processed book counts
    processed_path = Path("data/processed/book-counts.csv")
    mo.stop(
        not processed_path.exists(),
        mo.md(
            "⚠️ **Processed data not found.** "
            "Please run `make all` (or `uv run python -m bookstats.counts --combine ...`) "
            "to generate `data/processed/book-counts.csv`."
        ),
    )
    counts_df = pl.read_csv(processed_path)
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
def _(book_selector, compute_zipf_fit, counts_df, generate_zipf_chart, mo, pl):
    mo.stop(book_selector.value == "None", mo.md("No books available."))

    book_counts = counts_df.filter(pl.col("book") == book_selector.value)

    # Compute descriptive Zipf fit using deepened bookstats.zipf
    fit_result = compute_zipf_fit(book_counts, book_name=book_selector.value)

    stats = mo.hstack(
        [
            mo.stat(label="Total Words", value=f"{fit_result.total_words:,}"),
            mo.stat(label="Unique Vocabulary", value=f"{fit_result.unique_words:,}"),
            mo.stat(label="Fit Slope (β₁)", value=f"{fit_result.slope:.3f}"),
            mo.stat(label="R² (Determination)", value=f"{fit_result.r_squared:.3f}"),
        ]
    )

    zipf_chart = generate_zipf_chart(fit_result)

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
