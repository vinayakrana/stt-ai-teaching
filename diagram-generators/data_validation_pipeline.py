"""
Data Validation Pipeline Diagram
Week 2: Data Validation Lecture
Illustrates the stages of data validation from raw data to verified dataset
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_data_validation_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Validation pipeline stages
    stages = [
        ("Raw Data", 1.0, "#99ccff"),
        ("Inspection", 2.8, "#99ccff"),
        ("Schema\nValidation", 5.0, "#99ccff"),
        ("Cleaning", 7.5, "#99ccff"),
        ("Drift\nDetection", 10.0, "#99ccff"),
        ("Verified\nDataset", 12.8, "#99ff99")
    ]

    # Draw boxes and arrows
    for i, (label, x, color) in enumerate(stages):
        # Draw box
        box = FancyBboxPatch(
            (x - 0.65, 1.2), 1.3, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='#333333',
            linewidth=3 if i == len(stages) - 1 else 2,
            zorder=2
        )
        ax.add_patch(box)

        # Add text
        ax.text(x + 0.025, 1.5, label,
                ha='center', va='center',
                fontsize=12, fontweight='bold',
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
    ax.text(7, 2.5, 'Data Validation Pipeline',
            ha='center', va='center',
            fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'slides', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    fig = create_data_validation_pipeline()
    output_path = os.path.join(output_dir, 'data_validation_pipeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
