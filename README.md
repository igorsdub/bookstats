# bookstats

A cumulative research-project course analyzing word frequencies in classic literature from Project Gutenberg.

## Purpose

This project investigates word frequency distributions and tests descriptive Zipf's law fits across classic texts.

## Books Analyzed

| Gutenberg ID | Title | Author | Source URL |
|---|---|---|---|
| 00084 | Frankenstein; or, the Modern Prometheus | Mary Wollstonecraft Shelley | https://www.gutenberg.org/ebooks/84 |

### Filename Convention

Raw book text files are stored using the pattern:
`<five-digit-gutenberg-id>_<hyphenated-short-title>.txt`

The leading 5-digit number is the official Project Gutenberg ebook ID padded with zeros, followed by an underscore separator and a lowercase hyphenated short title.

| 00345 | Dracula | Bram Stoker | https://www.gutenberg.org/ebooks/345 |
| 01342 | Pride and Prejudice | Jane Austen | https://www.gutenberg.org/ebooks/1342 |

## Repository Structure

```text
├── data/
│   ├── raw/           # Original Project Gutenberg text files
│   └── intermediate/  # Generated per-book count tables (uncommitted)
├── src/
│   └── bookstats/     # Reusable analysis package
│       ├── __init__.py
│       └── counts.py
├── notebooks/
│   └── visualize.py   # Interactive Marimo visualization
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependencies
├── Makefile           # Automation DAG
└── README.md          # Project overview
```

## Setup Instructions

This project requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

To reconstruct the virtual environment:

```bash
uv sync
```

To select the environment in VS Code:
1. Open the Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Linux/WSL).
2. Run **Python: Select Interpreter**.
3. Choose the `.venv` interpreter (`.venv/bin/python`).


## Running the Analysis

Process a raw book using the `bookstats` package:

```bash
uv run python -m bookstats.counts data/raw/00084_frankenstein.txt data/intermediate/00084_frankenstein.csv
```

Combine all processed books:

```bash
uv run python -m bookstats.counts --combine data/intermediate/*.csv -o data/processed/book-counts.csv
```

Open the interactive Marimo visualization:

```bash
uv run marimo edit notebooks/visualize.py
```


## Reproducing Results

This project uses Make to automate the data pipeline and test checks.

Run linter and tests:

```bash
make check
```

Build all processed data, summary tables, and figures:

```bash
make all
```

Recreate all results from scratch:

```bash
make clean
make check
make all
```

