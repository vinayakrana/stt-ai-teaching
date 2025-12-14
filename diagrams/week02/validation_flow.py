"""
Generate data validation flow diagrams for Week 2: Data Validation
"""
from graphviz import Digraph

def generate_pydantic_validation_pipeline():
    """Create Pydantic validation pipeline diagram."""
    dot = Digraph('Validation Pipeline', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Define nodes
    dot.node('Input', 'Raw Input\n(JSON/Dict)', shape='parallelogram', style='filled', fillcolor='lightgray')
    dot.node('Schema', 'Pydantic\nSchema', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Valid', 'Validated Data\n✓', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Error', 'Validation Error\n✗', shape='box', style='filled', fillcolor='lightcoral')

    # Validation flow
    dot.edge('Input', 'Schema', label='validate')
    dot.edge('Schema', 'Valid', label='valid', color='green')
    dot.edge('Schema', 'Error', label='invalid', color='red')

    # Save diagram
    dot.render('../figures/week02_validation_pipeline', format='png', cleanup=True)
    print("Generated: figures/week02_validation_pipeline.png")

def generate_type_hierarchy():
    """Create Pydantic type hierarchy diagram."""
    dot = Digraph('Type Hierarchy', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Base model
    dot.node('BaseModel', 'BaseModel', shape='box', style='filled', fillcolor='lightyellow')

    # Child models
    dot.node('User', 'User', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Product', 'Product', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Order', 'Order', shape='box', style='filled', fillcolor='lightblue')

    # Inheritance
    dot.edge('BaseModel', 'User', label='extends')
    dot.edge('BaseModel', 'Product', label='extends')
    dot.edge('BaseModel', 'Order', label='extends')

    # Relations
    dot.node('OrderItem', 'OrderItem', shape='box', style='filled', fillcolor='lightgreen')
    dot.edge('Order', 'OrderItem', label='has many', style='dashed')
    dot.edge('OrderItem', 'Product', label='references', style='dashed')

    # Save diagram
    dot.render('../figures/week02_type_hierarchy', format='png', cleanup=True)
    print("Generated: figures/week02_type_hierarchy.png")

if __name__ == '__main__':
    print("Generating Week 2 diagrams...")
    generate_pydantic_validation_pipeline()
    generate_type_hierarchy()
    print("All Week 2 diagrams generated successfully!")
