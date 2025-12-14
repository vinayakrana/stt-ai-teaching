"""
Generate reproducibility and Docker diagrams for Week 8: Reproducibility
"""
from graphviz import Digraph

def generate_docker_architecture():
    """Create Docker architecture diagram."""
    dot = Digraph('Docker Architecture', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Dockerfile
    dot.node('Dockerfile', 'Dockerfile\n(Recipe)', shape='note', style='filled', fillcolor='lightyellow')

    # Build process
    dot.node('Build', 'docker build', shape='box', style='filled', fillcolor='lightblue')

    # Image
    dot.node('Image', 'Docker Image\n(Template)', shape='box', style='filled', fillcolor='lightgreen')

    # Run process
    dot.node('Run', 'docker run', shape='box', style='filled', fillcolor='lightblue')

    # Containers
    dot.node('Container1', 'Container 1\n(Instance)', shape='box3d', style='filled', fillcolor='lightcoral')
    dot.node('Container2', 'Container 2\n(Instance)', shape='box3d', style='filled', fillcolor='lightcoral')

    # Flow
    dot.edge('Dockerfile', 'Build')
    dot.edge('Build', 'Image')
    dot.edge('Image', 'Run')
    dot.edge('Run', 'Container1')
    dot.edge('Run', 'Container2')

    # Save diagram
    dot.render('../figures/week08_docker_architecture', format='png', cleanup=True)
    print("Generated: figures/week08_docker_architecture.png")

def generate_dvc_pipeline():
    """Create DVC pipeline diagram."""
    dot = Digraph('DVC Pipeline', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Data stages
    dot.node('Raw', 'Raw Data', shape='cylinder', style='filled', fillcolor='lightgray')
    dot.node('Prep', 'prepare.py\n(DVC stage)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Processed', 'Processed Data', shape='cylinder', style='filled', fillcolor='lightyellow')
    dot.node('Train', 'train.py\n(DVC stage)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Model', 'Model', shape='cylinder', style='filled', fillcolor='lightgreen')
    dot.node('Eval', 'evaluate.py\n(DVC stage)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Metrics', 'metrics.json', shape='note', style='filled', fillcolor='lightcoral')

    # Pipeline
    dot.edge('Raw', 'Prep', label='input')
    dot.edge('Prep', 'Processed', label='output')
    dot.edge('Processed', 'Train', label='input')
    dot.edge('Train', 'Model', label='output')
    dot.edge('Model', 'Eval', label='input')
    dot.edge('Eval', 'Metrics', label='output')

    # Save diagram
    dot.render('../figures/week08_dvc_pipeline', format='png', cleanup=True)
    print("Generated: figures/week08_dvc_pipeline.png")

def generate_mlflow_tracking():
    """Create MLflow tracking architecture diagram."""
    dot = Digraph('MLflow Tracking', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Training script
    dot.node('Script', 'train.py', shape='note', style='filled', fillcolor='lightblue')

    # MLflow components
    dot.node('Tracking', 'MLflow\nTracking Server', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('DB', 'Metadata DB\n(params, metrics)', shape='cylinder', style='filled', fillcolor='lightyellow')
    dot.node('Storage', 'Artifact Storage\n(models, plots)', shape='cylinder', style='filled', fillcolor='lightcoral')

    # UI
    dot.node('UI', 'MLflow UI\n(localhost:5000)', shape='box', style='filled', fillcolor='lightpink')

    # Connections
    dot.edge('Script', 'Tracking', label='log params,\nmetrics, artifacts')
    dot.edge('Tracking', 'DB', label='store metadata')
    dot.edge('Tracking', 'Storage', label='store artifacts')
    dot.edge('UI', 'Tracking', label='query')

    # Save diagram
    dot.render('../figures/week08_mlflow_tracking', format='png', cleanup=True)
    print("Generated: figures/week08_mlflow_tracking.png")

if __name__ == '__main__':
    print("Generating Week 8 diagrams...")
    generate_docker_architecture()
    generate_dvc_pipeline()
    generate_mlflow_tracking()
    print("All Week 8 diagrams generated successfully!")
