"""
Sync vs Async Request Timing Diagram
Week 1: Data Collection Lecture
Illustrates the time difference between synchronous and asynchronous request handling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

def create_sync_vs_async_timing():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Time range: 0 to 7 seconds
    time_max = 7

    # Synchronous requests
    sync_requests = [
        ("Request 1", 0, 2, '#3b82f6'),
        ("Request 2", 2, 4, '#8b5cf6'),
        ("Request 3", 4, 6, '#ec4899'),
    ]

    ax1.set_xlim(0, time_max)
    ax1.set_ylim(0, 1.5)
    ax1.set_title('Synchronous Requests (Sequential)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_yticks([])
    ax1.set_xlabel('')

    for i, (label, start, end, color) in enumerate(sync_requests):
        rect = Rectangle((start, 0.3), end - start, 0.6,
                        facecolor=color, edgecolor='#1e293b', linewidth=2, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text((start + end) / 2, 0.6, label,
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # Add total time annotation
    ax1.annotate('', xy=(6, 1.2), xytext=(0, 1.2),
                arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=2))
    ax1.text(3, 1.35, 'Total: 6 seconds', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#ef4444')

    # Asynchronous requests
    async_requests = [
        ("Request 1", 0.0, 2.0, '#3b82f6'),
        ("Request 2", 0.1, 2.1, '#8b5cf6'),
        ("Request 3", 0.2, 2.2, '#ec4899'),
    ]

    ax2.set_xlim(0, time_max)
    ax2.set_ylim(0, 2.5)
    ax2.set_title('Asynchronous Requests (Concurrent)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_yticks([])
    ax2.set_xlabel('Time (seconds)', fontsize=12)

    # Draw bars at different vertical positions
    for i, (label, start, end, color) in enumerate(async_requests):
        y_pos = 0.3 + i * 0.7
        rect = Rectangle((start, y_pos), end - start, 0.5,
                        facecolor=color, edgecolor='#1e293b', linewidth=2, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text((start + end) / 2, y_pos + 0.25, label,
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # Add total time annotation
    ax2.annotate('', xy=(2.2, 2.3), xytext=(0, 2.3),
                arrowprops=dict(arrowstyle='<->', color='#22c55e', lw=2))
    ax2.text(1.1, 2.45, 'Total: 2.2 seconds', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#22c55e')

    # Add grid for time reference
    for ax in [ax1, ax2]:
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xticks(range(0, time_max + 1))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'slides', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    fig = create_sync_vs_async_timing()
    output_path = os.path.join(output_dir, 'sync_vs_async_timing.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
