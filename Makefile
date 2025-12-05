.PHONY: all clean help list slides dark

# Find all .md files for Marp (excluding README.md)
MARP_SLIDES := $(filter-out README.md Readme.md, $(wildcard *.md))
MARP_PDF := $(MARP_SLIDES:.md=-marp.pdf)
MARP_PDF_DARK := $(MARP_SLIDES:.md=-marp-dark.pdf)
MARP_HTML := $(MARP_SLIDES:.md=-marp.html)

# Default target - build Marp slides
all: slides
	@echo "✓ All slides built successfully"

# Build Marp slides (PDF and HTML)
slides: $(MARP_PDF) $(MARP_HTML)
	@echo "✓ Marp slides built (PDF and HTML)"

# Build dark PDFs (inverted colors)
dark: $(MARP_PDF) $(MARP_PDF_DARK)
	@echo "✓ Dark theme PDFs created"

# Pattern rule for Marp PDF
%-marp.pdf: %.md
	@echo "Building Marp PDF: $< -> $@"
	@marp $< -o $@ --pdf --allow-local-files

# Pattern rule for Marp HTML
%-marp.html: %.md
	@echo "Building Marp HTML: $< -> $@"
	@marp $< -o $@ --html

# Pattern rule for dark PDF (color inversion with ghostscript)
%-marp-dark.pdf: %-marp.pdf
	@echo "Creating dark version: $< -> $@"
	@if command -v gs >/dev/null 2>&1; then \
		gs -o $@ -sDEVICE=pdfwrite \
		   -c "{1 exch sub}{1 exch sub}{1 exch sub}{1 exch sub} setcolortransfer" \
		   -f $< 2>&1 | grep -v "GPL Ghostscript" | grep -v "Copyright" | grep -v "Processing pages" | grep -v "Page [0-9]" | grep -v "reserved for an Annotation" || true; \
	else \
		echo "Warning: ghostscript (gs) not found. Copying original."; \
		cp $< $@; \
	fi

# List available slides
list:
	@echo "Available Marp slides:"
	@for file in $(MARP_SLIDES); do \
		echo "  - $${file%.md}"; \
	done
	@echo ""
	@echo "Build commands:"
	@echo "  make slides    # Build slides (PDF and HTML) - default"
	@echo "  make dark      # Build dark theme PDFs (inverted colors)"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(MARP_PDF) $(MARP_PDF_DARK) $(MARP_HTML)
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  make slides    # Build Marp slides (PDF and HTML) - default"
	@echo "  make dark      # Build dark theme PDFs (inverted colors)"
	@echo "  make list      # List available slides"
	@echo "  make clean     # Remove all generated files"
