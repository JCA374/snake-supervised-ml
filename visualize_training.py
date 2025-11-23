"""
Training visualization utilities.

Creates plots showing model improvement over training iterations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_progress(history, save_path='training_logs/training_progress.png',
                          human_baseline=None):
    """
    Plot training progress showing score improvement over iterations.

    Args:
        history: dict with keys 'iteration', 'mean_score', 'std_score', 'max_score'
        save_path: path to save the plot
        human_baseline: optional human baseline score to show as reference line

    Returns:
        path to saved plot
    """
    if not history or len(history['iteration']) == 0:
        print("No history to plot")
        return None

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = history['iteration']
    mean_scores = history['mean_score']
    std_scores = history['std_score']
    max_scores = history['max_score']

    # Plot 1: Mean score with error bars
    ax1.errorbar(iterations, mean_scores, yerr=std_scores,
                 marker='o', linewidth=2, markersize=8,
                 capsize=5, capthick=2, label='Mean Score ± Std')

    # Add human baseline if provided
    if human_baseline is not None:
        ax1.axhline(y=human_baseline, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Human Baseline ({human_baseline:.1f})')

    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress: Mean Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Annotate iteration 0 (imitation) and final iteration
    if len(iterations) > 0:
        # Iteration 0
        ax1.annotate(f'Imitation\n{mean_scores[0]:.1f}',
                    xy=(iterations[0], mean_scores[0]),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=9, fontweight='bold')

        # Final iteration
        if len(iterations) > 1:
            final_idx = len(iterations) - 1
            ax1.annotate(f'Final\n{mean_scores[final_idx]:.1f}',
                        xy=(iterations[final_idx], mean_scores[final_idx]),
                        xytext=(10, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9, fontweight='bold')

    # Plot 2: Max score progression
    ax2.plot(iterations, max_scores, marker='s', linewidth=2,
             markersize=8, color='green', label='Max Score')

    # Add human baseline if provided
    if human_baseline is not None:
        ax2.axhline(y=human_baseline, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Human Baseline ({human_baseline:.1f})')

    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Score', fontsize=12, fontweight='bold')
    ax2.set_title('Training Progress: Max Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Add improvement annotation on max score
    if len(iterations) > 1:
        improvement = max_scores[-1] - max_scores[0]
        ax2.text(0.05, 0.95, f'Total Improvement: {improvement:+.1f}',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training progress plot saved to {save_path}")

    plt.close()
    return save_path


def plot_comparison(history, save_path='training_logs/model_comparison.png',
                   human_baseline=None):
    """
    Create a bar chart comparing all model iterations.

    Args:
        history: dict with training history
        save_path: path to save the plot
        human_baseline: optional human baseline score

    Returns:
        path to saved plot
    """
    if not history or len(history['iteration']) == 0:
        print("No history to plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = history['iteration']
    mean_scores = history['mean_score']
    max_scores = history['max_score']

    x = np.arange(len(iterations))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, mean_scores, width, label='Mean Score',
                   color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, max_scores, width, label='Max Score',
                   color='lightgreen', edgecolor='black', linewidth=1.5)

    # Add human baseline
    if human_baseline is not None:
        ax.axhline(y=human_baseline, color='red', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Human Baseline ({human_baseline:.1f})')

    # Customize
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Iterations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)

    # Create labels (Imitation for iter 0, MC 1-N for rest)
    labels = ['Imitation'] + [f'MC {i}' for i in range(1, len(iterations))]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Model comparison plot saved to {save_path}")

    plt.close()
    return save_path


def create_training_report(history, human_baseline=None, videos=None,
                          save_path='training_logs/training_report.txt'):
    """
    Create a text report summarizing training progress.

    Args:
        history: dict with training history
        human_baseline: optional human baseline score
        videos: optional dict mapping iteration -> video_path
        save_path: path to save report

    Returns:
        path to saved report
    """
    if not history or len(history['iteration']) == 0:
        print("No history to report")
        return None

    lines = []
    lines.append("="*70)
    lines.append("SNAKE ML TRAINING REPORT")
    lines.append("="*70)
    lines.append("")

    # Human baseline
    if human_baseline is not None:
        lines.append(f"Human Baseline Score: {human_baseline:.2f}")
        lines.append("")

    # Training summary
    lines.append("Training Summary:")
    lines.append(f"  Total iterations: {len(history['iteration']) - 1}")  # -1 because iter 0 is imitation
    lines.append(f"  Initial mean score: {history['mean_score'][0]:.2f}")
    lines.append(f"  Final mean score: {history['mean_score'][-1]:.2f}")
    total_improvement = history['mean_score'][-1] - history['mean_score'][0]
    lines.append(f"  Total improvement: {total_improvement:+.2f}")
    lines.append("")

    # Best iteration
    best_idx = np.argmax(history['mean_score'])
    lines.append(f"Best Performance:")
    lines.append(f"  Iteration: {history['iteration'][best_idx]}")
    lines.append(f"  Mean score: {history['mean_score'][best_idx]:.2f} ± {history['std_score'][best_idx]:.2f}")
    lines.append(f"  Max score: {history['max_score'][best_idx]}")
    lines.append("")

    # Iteration details
    lines.append("Detailed Results:")
    lines.append("-"*70)
    lines.append(f"{'Iter':<8} {'Model':<15} {'Mean Score':<15} {'Max Score':<12} {'Improvement':<12}")
    lines.append("-"*70)

    for i, iter_num in enumerate(history['iteration']):
        model_name = "Imitation" if iter_num == 0 else f"MC Iter {iter_num}"
        improvement = 0 if i == 0 else history['mean_score'][i] - history['mean_score'][i-1]
        mean_score_str = f"{history['mean_score'][i]:.2f} ± {history['std_score'][i]:.2f}"

        lines.append(f"{iter_num:<8} {model_name:<15} {mean_score_str:<15} "
                    f"{history['max_score'][i]:<12} {improvement:+.2f}")

    lines.append("-"*70)
    lines.append("")

    # Video links
    if videos:
        lines.append("Recorded Videos:")
        for iter_num, video_path in sorted(videos.items()):
            model_name = "Imitation" if iter_num == 0 else f"MC Iter {iter_num}"
            lines.append(f"  {model_name}: {video_path}")
        lines.append("")

    lines.append("="*70)
    lines.append("End of Report")
    lines.append("="*70)

    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Training report saved to {save_path}")
    return save_path


if __name__ == '__main__':
    # Test visualization with dummy data
    history = {
        'iteration': [0, 1, 2, 3, 4, 5],
        'mean_score': [3.2, 4.5, 5.8, 6.2, 6.9, 7.1],
        'std_score': [1.2, 1.5, 1.3, 1.1, 0.9, 0.8],
        'max_score': [8, 12, 15, 18, 20, 22],
        'mean_length': [50, 70, 90, 100, 110, 115]
    }

    human_baseline = 5.0

    # Create plots
    plot_training_progress(history, human_baseline=human_baseline)
    plot_comparison(history, human_baseline=human_baseline)

    # Create report
    create_training_report(history, human_baseline=human_baseline)

    print("\n✓ Test visualizations created in training_logs/")
