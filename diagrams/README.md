# Lecture Diagrams Generation

This directory contains Python scripts to generate professional diagrams for all lecture slides.

## Prerequisites

Install required packages:

```bash
pip install graphviz
```

Also ensure Graphviz system package is installed:

**macOS**:
```bash
brew install graphviz
```

**Ubuntu/Debian**:
```bash
sudo apt-get install graphviz
```

**Windows**:
Download from: https://graphviz.org/download/

## Usage

### Generate All Diagrams

Run the master script to generate all diagrams at once:

```bash
cd diagrams
python generate_all.py
```

This will create all diagram PNG files in the `figures/` directory.

### Generate Diagrams for Specific Week

Run individual scripts:

```bash
cd diagrams/week01
python http_flow.py
```

## Directory Structure

```
diagrams/
├── week01/
│   └── http_flow.py           # HTTP flows, web scraping, API auth
├── week02/
│   └── validation_flow.py     # Pydantic validation pipeline
├── week03/
│   └── labeling_workflow.py   # Data labeling workflows
├── week04/
│   └── active_learning_loop.py # Active learning cycle
├── week05/
│   └── augmentation_pipeline.py # Data augmentation
├── week06/
│   └── llm_api_flow.py        # LLM API flows
├── week08/
│   └── docker_architecture.py  # Docker, DVC, MLflow
├── week09/
│   └── streamlit_execution.py  # Streamlit execution model
├── week11/
│   └── git_cicd_workflow.py    # Git branching, CI/CD
└── generate_all.py             # Master generation script
```

## Output

All generated diagrams are saved to:

```
figures/
├── week01_http_flow.png
├── week01_web_scraping.png
├── week01_api_auth.png
├── week02_validation_pipeline.png
├── week02_type_hierarchy.png
├── week03_labeling_workflow.png
├── week03_label_studio_arch.png
├── week04_active_learning_loop.png
├── week04_uncertainty_sampling.png
├── week05_augmentation_pipeline.png
├── week05_augmentation_comparison.png
├── week06_llm_api_flow.png
├── week06_prompt_patterns.png
├── week08_docker_architecture.png
├── week08_dvc_pipeline.png
├── week08_mlflow_tracking.png
├── week09_streamlit_rerun.png
├── week09_session_state.png
├── week09_deployment_architecture.png
├── week11_git_branching.png
├── week11_cicd_pipeline.png
└── week11_github_actions.png
```

## Using Diagrams in Slides

Reference the generated diagrams in your Marp slides:

```markdown
# HTTP Request/Response Flow

![HTTP Flow](../figures/week01_http_flow.png)
```

## Customization

To customize diagrams:

1. Edit the respective Python script
2. Modify colors, shapes, labels as needed
3. Re-run the script to regenerate

## Diagram Types by Week

- **Week 1**: HTTP flows, web scraping architecture, API authentication
- **Week 2**: Pydantic validation pipeline, type hierarchy
- **Week 3**: Data labeling workflow, Label Studio architecture
- **Week 4**: Active learning loop, uncertainty sampling
- **Week 5**: Augmentation pipeline, before/after comparison
- **Week 6**: LLM API request/response, prompt engineering patterns
- **Week 8**: Docker architecture, DVC pipeline, MLflow tracking
- **Week 9**: Streamlit re-run model, session state, deployment options
- **Week 11**: Git branching strategy, CI/CD pipeline, GitHub Actions

## Future Additions

Planned diagram conversions:
- Week 7: Convert Mermaid diagrams to graphviz
- Week 10: Convert ONNX Mermaid diagram
- Week 12: Convert model optimization Mermaid diagrams
- Week 13: Convert ASCII optimization loop to graphviz
- Week 14: Convert ASCII monitoring architecture to professional diagram
