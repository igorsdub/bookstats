"""bookstats: A Python package for analyzing word frequencies in Project Gutenberg books."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstats.counts import (
        combine_word_counts,
        count_words,
        extract_words,
        process_book_file,
        strip_gutenberg_headers,
    )

__version__ = "0.1.0"
__all__ = [
    "combine_word_counts",
    "count_words",
    "extract_words",
    "process_book_file",
    "strip_gutenberg_headers",
]


def __getattr__(name: str):
    if name in __all__:
        import bookstats.counts as _counts

        return getattr(_counts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
