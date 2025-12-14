"""
Master script to generate all lecture diagrams.

This script runs all individual diagram generation scripts to create
professional diagrams for the lecture slides using graphviz.

Usage:
    python generate_all.py
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Generate all diagrams."""
    diagrams_dir = Path(__file__).parent

    # List of all diagram generation scripts
    scripts = [
        diagrams_dir / 'week01' / 'http_flow.py',
        diagrams_dir / 'week02' / 'validation_flow.py',
        diagrams_dir / 'week03' / 'labeling_workflow.py',
        diagrams_dir / 'week04' / 'active_learning_loop.py',
        diagrams_dir / 'week05' / 'augmentation_pipeline.py',
        diagrams_dir / 'week06' / 'llm_api_flow.py',
        diagrams_dir / 'week07' / 'model_diagrams.py',
        diagrams_dir / 'week08' / 'docker_architecture.py',
        diagrams_dir / 'week09' / 'streamlit_execution.py',
        diagrams_dir / 'week11' / 'git_cicd_workflow.py',
        diagrams_dir / 'week12' / 'deployment_diagrams.py',
        diagrams_dir / 'week13' / 'optimization_loop.py',
        diagrams_dir / 'week14' / 'monitoring_architecture.py',
    ]

    print("="*60)
    print("GENERATING ALL LECTURE DIAGRAMS")
    print("="*60)

    success_count = 0
    fail_count = 0

    for script in scripts:
        if script.exists():
            if run_script(script):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"WARNING: Script not found: {script}")
            fail_count += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📁 Diagrams saved to: figures/")
    print("="*60)

    return fail_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
