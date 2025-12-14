"""
Generate HTTP request/response flow diagram for Week 1: Data Collection
"""
from graphviz import Digraph

def generate_http_flow():
    """Create HTTP request/response flow diagram."""
    dot = Digraph('HTTP Flow', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Define nodes
    dot.node('Client', 'Client\n(Browser/Script)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Server', 'Web Server', shape='box', style='filled', fillcolor='lightgreen')

    # Request flow
    dot.edge('Client', 'Server', label='HTTP Request\nGET /api/data', color='blue', fontcolor='blue')

    # Response flow
    dot.edge('Server', 'Client', label='HTTP Response\n200 OK + Data', color='green', fontcolor='green')

    # Save diagram
    dot.render('../figures/week01_http_flow', format='png', cleanup=True)
    print("Generated: figures/week01_http_flow.png")

def generate_web_scraping_architecture():
    """Create web scraping architecture diagram."""
    dot = Digraph('Web Scraping', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Define nodes with different shapes
    dot.node('Browser', 'Web Browser\n(requests)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('HTML', 'HTML Response', shape='note', style='filled', fillcolor='lightyellow')
    dot.node('Parser', 'HTML Parser\n(BeautifulSoup)', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Data', 'Structured Data\n(JSON/CSV)', shape='cylinder', style='filled', fillcolor='lightcoral')

    # Connections
    dot.edge('Browser', 'HTML', label='GET request')
    dot.edge('HTML', 'Parser', label='parse')
    dot.edge('Parser', 'Data', label='extract & save')

    # Save diagram
    dot.render('../figures/week01_web_scraping', format='png', cleanup=True)
    print("Generated: figures/week01_web_scraping.png")

def generate_api_auth_flow():
    """Create API authentication flow diagram."""
    dot = Digraph('API Auth', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Define nodes
    dot.node('Client', 'Client Application', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Auth', 'Auth Server', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('API', 'API Server', shape='box', style='filled', fillcolor='lightgreen')

    # Authentication flow
    dot.edge('Client', 'Auth', label='1. Login\n(username/password)', color='blue')
    dot.edge('Auth', 'Client', label='2. Access Token', color='green')
    dot.edge('Client', 'API', label='3. API Request\n(with token)', color='blue')
    dot.edge('API', 'Client', label='4. Data Response', color='green')

    # Save diagram
    dot.render('../figures/week01_api_auth', format='png', cleanup=True)
    print("Generated: figures/week01_api_auth.png")

if __name__ == '__main__':
    print("Generating Week 1 diagrams...")
    generate_http_flow()
    generate_web_scraping_architecture()
    generate_api_auth_flow()
    print("All Week 1 diagrams generated successfully!")
