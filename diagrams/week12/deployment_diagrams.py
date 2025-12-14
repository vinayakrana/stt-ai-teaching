"""
Generate deployment diagrams for Week 12: Deployment on Constrained Devices
Converts Mermaid diagrams to professional graphviz diagrams
"""
from graphviz import Digraph

def generate_model_optimization_techniques():
    """Create model optimization techniques diagram."""
    dot = Digraph('Model Optimization', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Trained model
    dot.node('Trained', 'Trained Model\n(FP32, Large)', shape='box', style='filled', fillcolor='lightblue')

    # Optimization techniques
    dot.node('Quant', 'Quantization\n(FP32→INT8)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Prune', 'Pruning\n(Remove weights)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Distill', 'Knowledge\nDistillation\n(Teacher→Student)', shape='box', style='filled', fillcolor='lightcoral')
    dot.node('NAS', 'Architecture\nSearch\n(Find efficient arch)', shape='box', style='filled', fillcolor='lightpink')

    # Optimized model
    dot.node('Optimized', 'Optimized Model\n(INT8, Smaller, Faster)', shape='box', style='filled', fillcolor='lightgreen')

    # Flow
    dot.edge('Trained', 'Quant')
    dot.edge('Trained', 'Prune')
    dot.edge('Trained', 'Distill')
    dot.edge('Trained', 'NAS')

    dot.edge('Quant', 'Optimized')
    dot.edge('Prune', 'Optimized')
    dot.edge('Distill', 'Optimized')
    dot.edge('NAS', 'Optimized')

    # Save diagram
    dot.render('../figures/week12_optimization_techniques', format='png', cleanup=True)
    print("Generated: figures/week12_optimization_techniques.png")

def generate_onnx_interoperability():
    """Create ONNX interoperability diagram."""
    dot = Digraph('ONNX Ecosystem', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Source frameworks
    dot.node('PyTorch', 'PyTorch', shape='box', style='filled', fillcolor='lightblue')
    dot.node('TF', 'TensorFlow', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('SKLearn', 'Scikit-Learn', shape='box', style='filled', fillcolor='lightgreen')

    # ONNX intermediate
    dot.node('ONNX', 'ONNX Graph\n(Intermediate\nRepresentation)', shape='ellipse', style='filled', fillcolor='lightcoral')

    # Runtime
    dot.node('ORT', 'ONNX Runtime', shape='box', style='filled', fillcolor='lightpink')

    # Deployment targets
    dot.node('Android', 'Android', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('RPi', 'Raspberry Pi', shape='box', style='filled', fillcolor='lightblue')
    dot.node('WASM', 'Browser\n(WebAssembly)', shape='box', style='filled', fillcolor='lightgreen')

    # Flow - Source to ONNX
    dot.edge('PyTorch', 'ONNX', label='export')
    dot.edge('TF', 'ONNX', label='export')
    dot.edge('SKLearn', 'ONNX', label='export')

    # ONNX to Runtime
    dot.edge('ONNX', 'ORT')

    # Runtime to targets
    dot.edge('ORT', 'Android', label='deploy')
    dot.edge('ORT', 'RPi', label='deploy')
    dot.edge('ORT', 'WASM', label='deploy')

    # Save diagram
    dot.render('../figures/week12_onnx_ecosystem', format='png', cleanup=True)
    print("Generated: figures/week12_onnx_ecosystem.png")

if __name__ == '__main__':
    print("Generating Week 12 diagrams...")
    generate_model_optimization_techniques()
    generate_onnx_interoperability()
    print("All Week 12 diagrams generated successfully!")
