"""
HTTP Request Sequence Diagram
Week 1: Data Collection Lecture
Illustrates the client-server interaction for an HTTP API request
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle
import os

def create_http_request_sequence():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Participants
    client_x = 2
    server_x = 8

    # Draw participant boxes
    for x, label in [(client_x, "Client\n(Browser/Python)"), (server_x, "Server\n(OMDb)")]:
        box = Rectangle((x - 0.8, 8.5), 1.6, 1.0,
                       facecolor='#e8f4f8',
                       edgecolor='#333333',
                       linewidth=2)
        ax.add_patch(box)
        ax.text(x, 9.0, label,
               ha='center', va='center',
               fontsize=12, fontweight='bold')

    # Draw lifelines
    for x in [client_x, server_x]:
        ax.plot([x, x], [0.5, 8.5], 'k--', linewidth=1.5, alpha=0.5)

    # Messages
    messages = [
        # (y, from_x, to_x, label, annotation, ann_side)
        (7.5, client_x, server_x, "HTTP Request\nGET /movie?t=Inception",
         "Headers: User-Agent, Auth", "left"),
        (5.5, server_x, server_x, "Process Request\n(Query DB)", "", "right"),
        (3.5, server_x, client_x, "HTTP Response\n(JSON Data)",
         "Status: 200 OK", "right"),
    ]

    for y, from_x, to_x, label, annotation, ann_side in messages:
        if from_x == to_x:
            # Self-call (processing)
            loop_width = 0.8
            loop = Rectangle((to_x, y - 0.3), loop_width, 0.6,
                           facecolor='#fff9e6',
                           edgecolor='#666666',
                           linewidth=1.5)
            ax.add_patch(loop)
            ax.text(to_x + loop_width/2, y, label,
                   ha='center', va='center',
                   fontsize=10, style='italic')
        else:
            # Regular message arrow
            is_return = to_x < from_x
            arrow_style = '->'
            line_style = 'dashed' if is_return else 'solid'

            arrow = FancyArrowPatch(
                (from_x, y), (to_x, y),
                arrowstyle=arrow_style,
                color='#2563eb' if not is_return else '#059669',
                linewidth=2,
                linestyle=line_style,
                mutation_scale=20
            )
            ax.add_patch(arrow)

            # Label above arrow
            mid_x = (from_x + to_x) / 2
            ax.text(mid_x, y + 0.3, label,
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

        # Annotation
        if annotation:
            ann_x = from_x - 1.5 if ann_side == "left" else to_x + 1.5
            ann_ha = 'right' if ann_side == "left" else 'left'
            ax.text(ann_x, y - 0.6, annotation,
                   ha=ann_ha, va='top',
                   fontsize=9, style='italic',
                   color='#64748b',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#cbd5e1'))

    # Title
    ax.text(5, 9.8, 'Client-Server HTTP Request Flow',
           ha='center', va='center',
           fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'slides', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    fig = create_http_request_sequence()
    output_path = os.path.join(output_dir, 'http_request_sequence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
