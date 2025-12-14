"""
Generate data augmentation diagrams for Week 5: Data Augmentation
"""
from graphviz import Digraph

def generate_augmentation_pipeline():
    """Create data augmentation pipeline diagram."""
    dot = Digraph('Augmentation Pipeline', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Original image
    dot.node('Original', 'Original Image', shape='box', style='filled', fillcolor='lightblue')

    # Augmentation techniques
    dot.node('Rotate', 'Rotate\n(±15°)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Flip', 'Horizontal Flip', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Crop', 'Random Crop', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Color', 'Color Jitter', shape='box', style='filled', fillcolor='lightyellow')

    # Augmented dataset
    dot.node('Augmented', 'Augmented Dataset\n(5x larger)', shape='cylinder', style='filled', fillcolor='lightgreen')

    # Pipeline
    dot.edge('Original', 'Rotate')
    dot.edge('Original', 'Flip')
    dot.edge('Original', 'Crop')
    dot.edge('Original', 'Color')

    dot.edge('Rotate', 'Augmented')
    dot.edge('Flip', 'Augmented')
    dot.edge('Crop', 'Augmented')
    dot.edge('Color', 'Augmented')

    # Save diagram
    dot.render('../figures/week05_augmentation_pipeline', format='png', cleanup=True)
    print("Generated: figures/week05_augmentation_pipeline.png")

def generate_augmentation_comparison():
    """Create before/after augmentation comparison diagram."""
    dot = Digraph('Augmentation Comparison', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Before
    dot.node('Before', 'Before Augmentation:\n\n1000 images\nLimited variety\nOverfitting risk: HIGH',
             shape='note', style='filled', fillcolor='lightcoral')

    # Augmentation
    dot.node('Aug', 'Data\nAugmentation', shape='box', style='filled', fillcolor='lightblue')

    # After
    dot.node('After', 'After Augmentation:\n\n5000 images\nHigh variety\nOverfitting risk: LOW',
             shape='note', style='filled', fillcolor='lightgreen')

    # Flow
    dot.edge('Before', 'Aug')
    dot.edge('Aug', 'After')

    # Save diagram
    dot.render('../figures/week05_augmentation_comparison', format='png', cleanup=True)
    print("Generated: figures/week05_augmentation_comparison.png")

if __name__ == '__main__':
    print("Generating Week 5 diagrams...")
    generate_augmentation_pipeline()
    generate_augmentation_comparison()
    print("All Week 5 diagrams generated successfully!")
