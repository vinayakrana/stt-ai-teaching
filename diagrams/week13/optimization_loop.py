"""
Generate optimization loop diagram for Week 13: Profiling & Optimization
Replaces ASCII art with professional graphviz diagram
"""
from graphviz import Digraph

def generate_optimization_loop():
    """Create professional optimization loop diagram."""
    dot = Digraph('Optimization Loop', graph_attr={'bgcolor': 'white'})

    # Define nodes
    dot.node('Baseline', 'Baseline Model', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Profile', 'Profile', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Identify', 'Identify\nBottleneck', shape='box', style='filled', fillcolor='lightcoral')
    dot.node('Optimize', 'Optimize\nSpecific\nComponent', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Measure', 'Measure\nImpact', shape='box', style='filled', fillcolor='lightpink')

    # Linear flow
    dot.edge('Baseline', 'Profile', label='1. Start')
    dot.edge('Profile', 'Identify', label='2. Analyze')
    dot.edge('Identify', 'Optimize', label='3. Fix')
    dot.edge('Optimize', 'Measure', label='4. Verify')

    # Feedback loop
    dot.edge('Measure', 'Profile', label='5. Repeat\nfor next\nbottleneck',
             color='blue', style='bold')

    # Save diagram
    dot.render('../figures/week13_optimization_loop', format='png', cleanup=True)
    print("Generated: figures/week13_optimization_loop.png")

if __name__ == '__main__':
    print("Generating Week 13 optimization loop diagram...")
    generate_optimization_loop()
    print("Week 13 diagram generated successfully!")
