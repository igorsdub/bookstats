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

## Repository Structure

```text
├── data/
│   ├── raw/           # Original Project Gutenberg text files
│   └── intermediate/  # Generated per-book count tables (uncommitted)
├── scripts/           # Executable analysis scripts
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependencies
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

Count words in a book:

```bash
uv run python scripts/count_words.py data/raw/00084_frankenstein.txt data/intermediate/00084_frankenstein.csv
```


## Reproducing Results

*(To be added in Lesson 4)*
