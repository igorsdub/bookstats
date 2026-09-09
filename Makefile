.POSIX:
.PHONY: all check clean help site

RAW_BOOKS = $(wildcard data/raw/*.txt)
INTERMEDIATE_COUNTS = $(patsubst data/raw/%.txt,data/intermediate/%.csv,$(RAW_BOOKS))
PROCESSED_DATA = data/processed/book-counts.csv
FIT_TABLE = output/tables/zipf-fits.csv
FIGURE_SVG = output/figures/zipf-law.svg
FIGURE_HTML = output/figures/zipf-law.html
SITE_INDEX = _site/index.html

# Default target: builds all analysis tables and figures
all: $(PROCESSED_DATA) $(FIT_TABLE) $(FIGURE_SVG) $(FIGURE_HTML)

# Run linter and automated test suite
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -v

# Pattern rule: Raw book -> Intermediate per-book word count CSV
data/intermediate/%.csv: data/raw/%.txt src/bookstats/counts.py
	@mkdir -p data/intermediate
	uv run python -m bookstats.counts $< $@

# Combined processed dataset: depends on all intermediate counts and counting code
$(PROCESSED_DATA): $(INTERMEDIATE_COUNTS) src/bookstats/counts.py
	@mkdir -p data/processed
	uv run python -m bookstats.counts --combine $(INTERMEDIATE_COUNTS) -o $@

# Reader-facing Zipf fit summary table
$(FIT_TABLE): $(PROCESSED_DATA) src/bookstats/zipf.py
	@mkdir -p output/tables
	uv run python -m bookstats.zipf $< --table $@

# Reader-facing figures: SVG and standalone interactive HTML
$(FIGURE_SVG) $(FIGURE_HTML): $(PROCESSED_DATA) src/bookstats/zipf.py
	@mkdir -p output/figures
	uv run python -m bookstats.zipf $< --svg $(FIGURE_SVG) --html $(FIGURE_HTML)

# Interactive Marimo application publication artifact
$(SITE_INDEX): $(PROCESSED_DATA) notebooks/visualize.py
	@mkdir -p _site
	uv run marimo export html notebooks/visualize.py -o $@

site: $(SITE_INDEX)

# Clean all generated, non-committed data, output artifacts, and website build
clean:
	rm -rf data/intermediate data/processed output _site .pytest_cache .ruff_cache

# Display available Make targets
help:
	@echo "Available make targets:"
	@echo "  all    - Build processed data, fit tables, and figures (default)"
	@echo "  check  - Run Ruff linting/formatting checks and pytest test suite"
	@echo "  clean  - Remove all generated data, outputs, and build artifacts"
	@echo "  site   - Export interactive Marimo website to _site/index.html"
	@echo "  help   - Show this help message"
