"""
Generate Git and CI/CD workflow diagrams for Week 11: Git & CI/CD
"""
from graphviz import Digraph

def generate_git_branching_strategy():
    """Create Git branching strategy diagram."""
    dot = Digraph('Git Branching', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Main branch
    dot.node('main1', 'main', shape='circle', style='filled', fillcolor='lightgreen')
    dot.node('main2', 'main', shape='circle', style='filled', fillcolor='lightgreen')
    dot.node('main3', 'main', shape='circle', style='filled', fillcolor='lightgreen')

    # Feature branch
    dot.node('feat1', 'feature\nbranch', shape='circle', style='filled', fillcolor='lightblue')
    dot.node('feat2', 'feature\nbranch', shape='circle', style='filled', fillcolor='lightblue')

    # Pull request
    dot.node('pr', 'Pull Request\n(Code Review)', shape='diamond', style='filled', fillcolor='lightyellow')

    # Main branch timeline
    dot.edge('main1', 'main2', label='commit')
    dot.edge('main2', 'main3', label='merge', color='green', style='bold')

    # Feature branch
    dot.edge('main1', 'feat1', label='branch off', style='dashed')
    dot.edge('feat1', 'feat2', label='commit')
    dot.edge('feat2', 'pr', label='create PR')
    dot.edge('pr', 'main3', label='approved', color='green')

    # Save diagram
    dot.render('../figures/week11_git_branching', format='png', cleanup=True)
    print("Generated: figures/week11_git_branching.png")

def generate_cicd_pipeline():
    """Create CI/CD pipeline flow diagram."""
    dot = Digraph('CI/CD Pipeline', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Code change
    dot.node('Push', 'git push', shape='box', style='filled', fillcolor='lightblue')

    # CI stages
    dot.node('Trigger', 'Trigger\nGitHub Actions', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Test', 'Run Tests\npytest', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Lint', 'Run Linters\nblack, mypy', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Build', 'Build Docker\nImage', shape='box', style='filled', fillcolor='lightgreen')

    # Decision
    dot.node('Pass', 'All Checks\nPass?', shape='diamond', style='filled', fillcolor='lightcoral')

    # Deployment
    dot.node('Deploy', 'Deploy to\nProduction', shape='box', style='filled', fillcolor='lightpink')
    dot.node('Fail', 'Notify\nFailure ❌', shape='box', style='filled', fillcolor='red')

    # Flow
    dot.edge('Push', 'Trigger')
    dot.edge('Trigger', 'Test', label='parallel')
    dot.edge('Trigger', 'Lint', label='parallel')
    dot.edge('Trigger', 'Build', label='parallel')

    dot.edge('Test', 'Pass')
    dot.edge('Lint', 'Pass')
    dot.edge('Build', 'Pass')

    dot.edge('Pass', 'Deploy', label='yes ✓', color='green')
    dot.edge('Pass', 'Fail', label='no ✗', color='red')

    # Save diagram
    dot.render('../figures/week11_cicd_pipeline', format='png', cleanup=True)
    print("Generated: figures/week11_cicd_pipeline.png")

def generate_github_actions_workflow():
    """Create GitHub Actions workflow visualization."""
    dot = Digraph('GitHub Actions', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Trigger
    dot.node('Event', 'Event Trigger\n(push, PR, schedule)', shape='ellipse', style='filled', fillcolor='lightblue')

    # Workflow file
    dot.node('Workflow', '.github/workflows/\ntrain.yml', shape='note', style='filled', fillcolor='lightyellow')

    # Jobs
    dot.node('Job1', 'Job: test\n- Setup Python\n- Install deps\n- Run pytest',
             shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Job2', 'Job: deploy\n- Build Docker\n- Push to registry\n- Deploy',
             shape='box', style='filled', fillcolor='lightcoral')

    # Flow
    dot.edge('Event', 'Workflow')
    dot.edge('Workflow', 'Job1')
    dot.edge('Workflow', 'Job2')
    dot.edge('Job1', 'Job2', label='depends_on', style='dashed')

    # Save diagram
    dot.render('../figures/week11_github_actions', format='png', cleanup=True)
    print("Generated: figures/week11_github_actions.png")

if __name__ == '__main__':
    print("Generating Week 11 diagrams...")
    generate_git_branching_strategy()
    generate_cicd_pipeline()
    generate_github_actions_workflow()
    print("All Week 11 diagrams generated successfully!")
