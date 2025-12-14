"""
Generate data labeling workflow diagrams for Week 3: Data Labeling
"""
from graphviz import Digraph

def generate_labeling_workflow():
    """Create data labeling workflow diagram."""
    dot = Digraph('Labeling Workflow', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Define nodes
    dot.node('Raw', 'Raw Data', shape='cylinder', style='filled', fillcolor='lightgray')
    dot.node('Annotators', 'Annotators', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Review', 'Quality Review', shape='diamond', style='filled', fillcolor='lightyellow')
    dot.node('Final', 'Labeled Dataset', shape='cylinder', style='filled', fillcolor='lightgreen')
    dot.node('Reject', 'Relabel', shape='box', style='filled', fillcolor='lightcoral')

    # Workflow
    dot.edge('Raw', 'Annotators', label='assign tasks')
    dot.edge('Annotators', 'Review', label='submit labels')
    dot.edge('Review', 'Final', label='approved', color='green')
    dot.edge('Review', 'Reject', label='rejected', color='red')
    dot.edge('Reject', 'Annotators', label='reassign', style='dashed')

    # Save diagram
    dot.render('../figures/week03_labeling_workflow', format='png', cleanup=True)
    print("Generated: figures/week03_labeling_workflow.png")

def generate_label_studio_architecture():
    """Create Label Studio architecture diagram."""
    dot = Digraph('Label Studio', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Components
    dot.node('UI', 'Web UI\n(Annotation Interface)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Backend', 'Label Studio\nBackend', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('DB', 'Database\n(Labels)', shape='cylinder', style='filled', fillcolor='lightyellow')
    dot.node('Storage', 'Storage\n(Raw Data)', shape='cylinder', style='filled', fillcolor='lightcoral')

    # Connections
    dot.edge('UI', 'Backend', label='HTTP', dir='both')
    dot.edge('Backend', 'DB', label='save labels')
    dot.edge('Backend', 'Storage', label='load data')

    # Save diagram
    dot.render('../figures/week03_label_studio_arch', format='png', cleanup=True)
    print("Generated: figures/week03_label_studio_arch.png")

if __name__ == '__main__':
    print("Generating Week 3 diagrams...")
    generate_labeling_workflow()
    generate_label_studio_architecture()
    print("All Week 3 diagrams generated successfully!")
