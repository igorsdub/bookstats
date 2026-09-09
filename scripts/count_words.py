"""Count word frequencies in a plain-text book and save results to CSV.

Usage:
    python count_words.py INPUT OUTPUT

Example:
    python count_words.py 00084_frankenstein.txt 00084_frankenstein.csv
"""

import re
import sys
from pathlib import Path
import polars as pl


def strip_gutenberg_headers(text: str) -> str:
    """Strip Project Gutenberg header and footer licenses from text if present."""
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    if start_match:
        text = text[start_match.end():]
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if end_match:
        text = text[:end_match.start()]
    return text


def extract_words(text: str) -> list[str]:
    """Extract and normalize lowercase words from a text string."""
    cleaned = strip_gutenberg_headers(text)
    return re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())


def count_words(words: list[str]) -> pl.DataFrame:
    """Count occurrences of each word and sort by frequency descending."""
    if not words:
        return pl.DataFrame({"word": [], "count": []}, schema={"word": pl.String, "count": pl.UInt32})
    df = pl.DataFrame({"word": words})
    return (
        df.group_by("word")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    words = extract_words(text)
    counts = count_words(words)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts.write_csv(output_path)
    print(f"Wrote {len(counts)} unique word counts to {output_path}")


if __name__ == "__main__":
    main()
