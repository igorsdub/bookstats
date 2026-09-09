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
    from bookstats.zipf import (
        ZipfFitResult,
        compute_zipf_fit,
        fit_all_books,
        generate_zipf_chart,
        plot_zipf,
    )

__version__ = "0.1.0"
__all__ = [
    "ZipfFitResult",
    "combine_word_counts",
    "compute_zipf_fit",
    "count_words",
    "extract_words",
    "fit_all_books",
    "generate_zipf_chart",
    "plot_zipf",
    "process_book_file",
    "strip_gutenberg_headers",
]


def __getattr__(name: str):
    if name in __all__:
        if name in [
            "ZipfFitResult",
            "compute_zipf_fit",
            "fit_all_books",
            "generate_zipf_chart",
            "plot_zipf",
        ]:
            import bookstats.zipf as _zipf

            return getattr(_zipf, name)
        import bookstats.counts as _counts

        return getattr(_counts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
