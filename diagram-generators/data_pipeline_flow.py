"""
Data Pipeline Flow Diagram
Week 1: Data Collection Lecture
Illustrates the ML data pipeline stages from collection to deployment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_data_pipeline_flow():
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Pipeline stages
    stages = [
        ("Collection", 1.5, "#ff9966"),
        ("Validation", 4.0, "#99ccff"),
        ("Labeling", 6.5, "#99ccff"),
        ("Training", 9.0, "#99ccff"),
        ("Deployment", 11.5, "#99ccff")
    ]

    # Draw boxes and arrows
    for i, (label, x, color) in enumerate(stages):
        # Draw box
        box = FancyBboxPatch(
            (x - 0.6, 1.2), 1.2, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='#333333',
            linewidth=3 if i == 0 else 2,
            zorder=2
        )
        ax.add_patch(box)

        # Add text
        ax.text(x, 1.5, label,
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                zorder=3)

        # Draw arrow to next stage
        if i < len(stages) - 1:
            next_x = stages[i + 1][1]
            arrow = FancyArrowPatch(
                (x + 0.7, 1.5), (next_x - 0.7, 1.5),
                arrowstyle='->,head_width=0.4,head_length=0.3',
                color='#333333',
                linewidth=2,
                zorder=1
            )
            ax.add_patch(arrow)

    # Add title
    ax.text(7, 2.5, 'ML Data Pipeline',
            ha='center', va='center',
            fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'slides', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    fig = create_data_pipeline_flow()
    output_path = os.path.join(output_dir, 'data_pipeline_flow.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
