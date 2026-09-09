import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import altair as alt
    import polars as pl
    from pathlib import Path

    return mo, alt, pl, Path


@app.cell
def __(mo):
    mo.md(
        r"""
        # Book Word Frequency Analysis

        An interactive visualization of word frequency distributions across Project Gutenberg books.
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
    books = sorted(counts_df["book"].unique().to_list()) if len(counts_df) > 0 else ["None"]
    book_selector = mo.ui.dropdown(
        options=books,
        value=books[0] if books else "None",
        label="Select a book:",
    )
    book_selector
    return book_selector, books


@app.cell
def __(alt, book_selector, counts_df, mo, pl):
    mo.stop(book_selector.value == "None", mo.md("No books available."))

    selected_df = (
        counts_df.filter(pl.col("book") == book_selector.value)
        .sort("count", descending=True)
        .head(30)
    )

    chart = (
        alt.Chart(selected_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Frequency Count"),
            y=alt.Y("word:N", sort="-x", title="Word"),
            tooltip=["word", "count"],
        )
        .properties(
            title=f"Top 30 Most Frequent Words: {book_selector.value}",
            width=600,
            height=500,
        )
    )

    mo.ui.altair_chart(chart)
    return chart, selected_df


if __name__ == "__main__":
    app.run()
