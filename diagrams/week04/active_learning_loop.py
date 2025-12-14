"""
Generate active learning diagrams for Week 4: Active Learning
"""
from graphviz import Digraph

def generate_active_learning_loop():
    """Create active learning cycle diagram."""
    dot = Digraph('Active Learning Loop', graph_attr={'bgcolor': 'white'})

    # Define nodes
    dot.node('Labeled', 'Labeled Data', shape='cylinder', style='filled', fillcolor='lightgreen')
    dot.node('Train', 'Train Model', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Predict', 'Predict on\nUnlabeled Pool', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Select', 'Select Most\nUncertain', shape='diamond', style='filled', fillcolor='lightcoral')
    dot.node('Label', 'Human\nAnnotation', shape='box', style='filled', fillcolor='lightpink')
    dot.node('Unlabeled', 'Unlabeled Pool', shape='cylinder', style='filled', fillcolor='lightgray')

    # Cycle
    dot.edge('Labeled', 'Train', label='1')
    dot.edge('Train', 'Predict', label='2')
    dot.edge('Unlabeled', 'Predict')
    dot.edge('Predict', 'Select', label='3')
    dot.edge('Select', 'Label', label='4')
    dot.edge('Label', 'Labeled', label='5. Add to\ntraining set', color='green', style='bold')

    # Save diagram
    dot.render('../figures/week04_active_learning_loop', format='png', cleanup=True)
    print("Generated: figures/week04_active_learning_loop.png")

def generate_uncertainty_sampling():
    """Create uncertainty sampling visualization diagram."""
    dot = Digraph('Uncertainty Sampling', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Model predictions
    dot.node('Model', 'Trained Model', shape='box', style='filled', fillcolor='lightblue')

    # Examples with different uncertainty levels
    dot.node('Certain', 'Certain\nP(cat)=0.95\n✓ Skip', shape='note', style='filled', fillcolor='lightgreen')
    dot.node('Uncertain', 'Uncertain\nP(cat)=0.51\n⚠ Label This!', shape='note', style='filled', fillcolor='lightcoral')
    dot.node('VeryUncertain', 'Very Uncertain\nP(cat)=0.50\n⚠⚠ Label First!', shape='note', style='filled', fillcolor='red')

    # Connections
    dot.edge('Model', 'Certain', label='high confidence')
    dot.edge('Model', 'Uncertain', label='low confidence')
    dot.edge('Model', 'VeryUncertain', label='very low confidence')

    # Save diagram
    dot.render('../figures/week04_uncertainty_sampling', format='png', cleanup=True)
    print("Generated: figures/week04_uncertainty_sampling.png")

if __name__ == '__main__':
    print("Generating Week 4 diagrams...")
    generate_active_learning_loop()
    generate_uncertainty_sampling()
    print("All Week 4 diagrams generated successfully!")
