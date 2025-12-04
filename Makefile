.PHONY: all clean help pdf list

# Find all .typ files (excluding slides.typ and other partials)
SLIDES := $(filter-out slides.typ, $(wildcard *.typ))
PDF_FILES := $(SLIDES:.typ=.pdf)

# Default target
all: pdf
	@echo "✓ All slides built successfully"

# Build all PDF slides
pdf: $(PDF_FILES)
	@echo "✓ PDF slides built"

# Pattern rule for PDF slides
%.pdf: %.typ slides.typ
	@echo "Compiling $< -> $@"
	@typst compile $< $@

# List available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES); do \
		echo "  - $${file%.typ}"; \
	done

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(PDF_FILES)
	@echo "✓ Clean complete"

help:
	@echo "Usage:"
	@echo "  make all       # Build all PDF slides"
	@echo "  make pdf       # Build all PDF slides"
	@echo "  make <name>    # Build specific slide (e.g., make data-collection-labeling.pdf)"
	@echo "  make list      # List available slides"
	@echo "  make clean     # Clean all artifacts"
