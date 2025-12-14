"""
Generate LLM API flow diagrams for Week 6: LLM APIs
"""
from graphviz import Digraph

def generate_api_request_response():
    """Create LLM API request/response flow diagram."""
    dot = Digraph('LLM API Flow', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Components
    dot.node('App', 'Your Application', shape='box', style='filled', fillcolor='lightblue')
    dot.node('API', 'LLM API\n(OpenAI/Anthropic)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Model', 'Language Model\n(GPT-4/Claude)', shape='ellipse', style='filled', fillcolor='lightyellow')

    # Request flow
    dot.edge('App', 'API', label='POST /chat/completions\n{\n  "prompt": "...",\n  "max_tokens": 100\n}',
             color='blue', fontcolor='blue')
    dot.edge('API', 'Model', label='forward prompt')

    # Response flow
    dot.edge('Model', 'API', label='generate text')
    dot.edge('API', 'App', label='200 OK\n{\n  "choices": [{\n    "text": "..."\n  }]\n}',
             color='green', fontcolor='green')

    # Save diagram
    dot.render('../figures/week06_llm_api_flow', format='png', cleanup=True)
    print("Generated: figures/week06_llm_api_flow.png")

def generate_prompt_engineering_patterns():
    """Create prompt engineering patterns diagram."""
    dot = Digraph('Prompt Engineering', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # User input
    dot.node('User', 'User Query', shape='parallelogram', style='filled', fillcolor='lightgray')

    # Pattern types
    dot.node('ZeroShot', 'Zero-Shot\n"Translate to French"', shape='box', style='filled', fillcolor='lightblue')
    dot.node('FewShot', 'Few-Shot\n"Example 1...\nExample 2...\nNow you:"', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('COT', 'Chain-of-Thought\n"Let\'s think step by step"', shape='box', style='filled', fillcolor='lightgreen')

    # LLM
    dot.node('LLM', 'LLM', shape='ellipse', style='filled', fillcolor='lightcoral')

    # Connections
    dot.edge('User', 'ZeroShot')
    dot.edge('User', 'FewShot')
    dot.edge('User', 'COT')

    dot.edge('ZeroShot', 'LLM')
    dot.edge('FewShot', 'LLM')
    dot.edge('COT', 'LLM')

    # Save diagram
    dot.render('../figures/week06_prompt_patterns', format='png', cleanup=True)
    print("Generated: figures/week06_prompt_patterns.png")

if __name__ == '__main__':
    print("Generating Week 6 diagrams...")
    generate_api_request_response()
    generate_prompt_engineering_patterns()
    print("All Week 6 diagrams generated successfully!")
