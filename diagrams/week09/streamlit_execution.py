"""
Generate interactive demo diagrams for Week 9: Interactive Demos
"""
from graphviz import Digraph

def generate_streamlit_rerun_model():
    """Create Streamlit re-run execution model diagram."""
    dot = Digraph('Streamlit Re-run Model', graph_attr={'bgcolor': 'white'})

    # User actions
    dot.node('User', 'User\nInteraction', shape='ellipse', style='filled', fillcolor='lightblue')

    # Script execution
    dot.node('Load', 'Load Page', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Run1', 'Run Script\n(Top to Bottom)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Display1', 'Display UI', shape='box', style='filled', fillcolor='lightcoral')

    # Re-run on interaction
    dot.node('Button', 'Button Click', shape='box', style='filled', fillcolor='lightpink')
    dot.node('Rerun', 'Re-run ENTIRE Script\n(Top to Bottom)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Display2', 'Update UI', shape='box', style='filled', fillcolor='lightcoral')

    # Flow
    dot.edge('User', 'Load')
    dot.edge('Load', 'Run1', label='1. First load')
    dot.edge('Run1', 'Display1')
    dot.edge('Display1', 'Button')
    dot.edge('Button', 'Rerun', label='2. On interaction')
    dot.edge('Rerun', 'Display2')
    dot.edge('Display2', 'Button', label='repeat', style='dashed')

    # Save diagram
    dot.render('../figures/week09_streamlit_rerun', format='png', cleanup=True)
    print("Generated: figures/week09_streamlit_rerun.png")

def generate_session_state_lifecycle():
    """Create session state lifecycle diagram."""
    dot = Digraph('Session State', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Initial state
    dot.node('NoState', 'No Session State\ncount = 0\n❌ Resets on rerun',
             shape='note', style='filled', fillcolor='lightcoral')

    # With session state
    dot.node('WithState', 'With Session State\nst.session_state.count\n✅ Persists across reruns',
             shape='note', style='filled', fillcolor='lightgreen')

    # Problem -> Solution
    dot.edge('NoState', 'WithState', label='Use st.session_state', color='green', style='bold')

    # Save diagram
    dot.render('../figures/week09_session_state', format='png', cleanup=True)
    print("Generated: figures/week09_session_state.png")

def generate_deployment_architecture():
    """Create demo deployment architecture diagram."""
    dot = Digraph('Deployment', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Development
    dot.node('Local', 'Local Development\nstreamlit run app.py', shape='box', style='filled', fillcolor='lightblue')

    # Deployment options
    dot.node('HF', 'Hugging Face Spaces\n(Free, GPU option)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('SC', 'Streamlit Cloud\n(Free tier)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Cloud', 'AWS/GCP/Azure\n(Custom setup)', shape='box', style='filled', fillcolor='lightcoral')

    # Users
    dot.node('Users', 'Users\n(Web Browser)', shape='ellipse', style='filled', fillcolor='lightpink')

    # Deployment flow
    dot.edge('Local', 'HF', label='git push')
    dot.edge('Local', 'SC', label='git push')
    dot.edge('Local', 'Cloud', label='docker deploy')

    # Access
    dot.edge('HF', 'Users', label='HTTPS', dir='both')
    dot.edge('SC', 'Users', label='HTTPS', dir='both')
    dot.edge('Cloud', 'Users', label='HTTPS', dir='both')

    # Save diagram
    dot.render('../figures/week09_deployment_architecture', format='png', cleanup=True)
    print("Generated: figures/week09_deployment_architecture.png")

if __name__ == '__main__':
    print("Generating Week 9 diagrams...")
    generate_streamlit_rerun_model()
    generate_session_state_lifecycle()
    generate_deployment_architecture()
    print("All Week 9 diagrams generated successfully!")
