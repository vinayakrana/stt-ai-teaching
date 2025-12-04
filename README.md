# CS 203: Software Tools and Techniques for AI

Course materials for CS 203 at IIT Gandhinagar.

**Course website:** https://nipunbatra.github.io/stt-ai-26/

**Slides (GitHub Pages):** https://nipunbatra.github.io/stt-ai-teaching/

## Slides

### Data Collection and Labeling

Comprehensive slides covering:
- Data Collection (instrumentation, analytics, logging, scraping, streaming)
- Data Validation (Pydantic, Great Expectations, Pandera, quality monitoring)
- Data Labeling (Label Studio, inter-annotator agreement, active learning, weak supervision)
- Data Augmentation (image, text, audio, time series, SMOTE, generative models)

**Files:**
- Source: `data-collection-labeling.qmd`
- PDF: `data-collection-labeling.pdf`

## Building Slides

### Prerequisites

Install Quarto: https://quarto.org/docs/get-started/

### Using Makefile (Recommended)

```bash
# Build entire site with Quarto (HTML + PDF)
make all

# Or use Quarto directly
quarto render

# Build only PDF documents
make pdf

# Build specific slide
make data-collection-labeling

# List available slides
make list

# Clean generated files
make clean
```

### Manual Build

**Render HTML Slides:**
```bash
quarto render data-collection-labeling.qmd --to revealjs
```

**Export to PDF:**
```bash
quarto render data-collection-labeling.qmd --to pdf
```

**Navigation (HTML slides):**
- Arrow keys / Space: Next slide
- `F`: Fullscreen
- `S`: Speaker notes
- `C`: Chalkboard (draw on slides)

### GitHub Actions (Automatic)

GitHub Actions automatically builds and deploys on every push.

**Workflow:**
1. Edit `.qmd` files
2. Build PDFs locally: `make pdf` (optional, for local preview)
3. Commit and push: `git add *.qmd *.pdf && git commit && git push`
4. GitHub Actions: Builds HTML and deploys to Pages automatically

**What's committed:** Only source files (`.qmd`) and PDFs
**What's generated:** HTML files are built by GitHub Actions

## Editing Slides

Edit the `.qmd` file directly - it's Markdown with special syntax for slides.

**Slide breaks:**
- `#` - New section (title slide)
- `##` - New slide

**Incremental lists:**
```markdown
::: incremental
- Item 1
- Item 2
:::
```

**Code blocks:**
````markdown
```python
# Your code here
```
````

**Columns:**
```markdown
::: columns
::: {.column width="50%"}
Left content
:::
::: {.column width="50%"}
Right content
:::
:::
```

## License

Course materials by Prof. Nipun Batra, IIT Gandhinagar.
