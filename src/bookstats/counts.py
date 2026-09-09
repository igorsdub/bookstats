"""Word extraction and frequency counting functions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl


def strip_gutenberg_headers(text: str) -> str:
    """Strip Project Gutenberg header and footer licenses from text.

    Parameters
    ----------
    text : str
        Raw text content of a Project Gutenberg book.

    Returns
    -------
    str
        Text content with license headers and footers removed.
    """
    start_match = re.search(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text
    )
    if start_match:
        text = text[start_match.end() :]
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if end_match:
        text = text[: end_match.start()]
    return text


def extract_words(text: str) -> list[str]:
    """Extract and normalize lowercase words from text.

    Parameters
    ----------
    text : str
        Input text to extract words from.

    Returns
    -------
    list of str
        List of lowercased word tokens with punctuation removed.
    """
    cleaned = strip_gutenberg_headers(text)
    return re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())


def count_words(words: list[str]) -> pl.DataFrame:
    """Count occurrences of each word and sort by frequency descending.

    Parameters
    ----------
    words : list of str
        List of normalized words.

    Returns
    -------
    polars.DataFrame
        DataFrame with columns 'word' and 'count', ordered from most
        frequent to least frequent.
    """
    if not words:
        return pl.DataFrame(
            {"word": [], "count": []},
            schema={"word": pl.String, "count": pl.UInt32},
        )
    df = pl.DataFrame({"word": words})
    return (
        df.group_by("word").agg(pl.len().alias("count")).sort("count", descending=True)
    )


def process_book_file(input_path: Path | str, output_path: Path | str) -> pl.DataFrame:
    """Process a single book text file and save word counts to CSV.

    Parameters
    ----------
    input_path : Path or str
        Path to the raw text input file.
    output_path : Path or str
        Destination path for the intermediate count CSV.

    Returns
    -------
    polars.DataFrame
        DataFrame of word counts that was written to disk.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    text = in_p.read_text(encoding="utf-8")
    words = extract_words(text)
    counts = count_words(words)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    counts.write_csv(out_p)
    return counts


def combine_word_counts(
    input_paths: list[Path | str], output_path: Path | str
) -> pl.DataFrame:
    """Combine multiple per-book count CSV files into a single dataset.

    Parameters
    ----------
    input_paths : list of (Path or str)
        Paths to the per-book intermediate count CSV files.
    output_path : Path or str
        Destination path for the processed combined CSV.

    Returns
    -------
    polars.DataFrame
        Combined DataFrame containing 'book', 'word', and 'count' columns.
    """
    frames = []
    for p in input_paths:
        path = Path(p)
        book_id = path.stem
        df = pl.read_csv(path)
        df = df.with_columns(pl.lit(book_id).alias("book"))
        df = df.select(["book", "word", "count"])
        frames.append(df)

    if frames:
        combined = pl.concat(frames)
    else:
        combined = pl.DataFrame(
            {"book": [], "word": [], "count": []},
            schema={"book": pl.String, "word": pl.String, "count": pl.UInt32},
        )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(out_p)
    return combined


def main() -> None:
    """Command-line interface for word counting."""
    parser = argparse.ArgumentParser(
        description="Count word frequencies in Project Gutenberg books."
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine multiple intermediate count CSVs into one table.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input text file (single mode) or CSV files (combine mode).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    if args.combine:
        if not args.output:
            print(
                "Error: --output is required when using --combine.",
                file=sys.stderr,
            )
            sys.exit(1)
        combine_word_counts(args.inputs, args.output)
        print(f"Combined {len(args.inputs)} books into {args.output}")
    else:
        if len(args.inputs) == 2 and not args.output:
            in_file, out_file = args.inputs[0], args.inputs[1]
        elif len(args.inputs) == 1 and args.output:
            in_file, out_file = args.inputs[0], args.output
        else:
            print(
                "Usage: python -m bookstats.counts INPUT OUTPUT",
                file=sys.stderr,
            )
            sys.exit(1)
        process_book_file(in_file, out_file)
        print(f"Processed {in_file} -> {out_file}")


if __name__ == "__main__":
    main()
