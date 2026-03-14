"""
utils/plotting.py

Training visualisation for LM-JEPA symbolic regression.
Saves annotated plots to results/plots/ after training.

Plots generated:
  1. Loss curves     — train/val loss over epochs
  2. JEPA breakdown  — LM vs JEPA vs SIGReg loss components
  3. Learning rate   — cosine annealing schedule
  4. Collapse monitor — embedding variance and cosine similarity
  5. Summary card    — single figure with all key metrics
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Style ─────────────────────────────────────────────────────────────────
STYLE = {
    'train_color':  '#2196F3',   # blue
    'val_color':    '#F44336',   # red
    'lm_color':     '#4CAF50',   # green
    'jepa_color':   '#FF9800',   # orange
    'sig_color':    '#9C27B0',   # purple
    'lr_color':     '#607D8B',   # grey
    'var_color':    '#00BCD4',   # cyan
    'cos_color':    '#FF5722',   # deep orange
    'bg_color':     '#FAFAFA',
    'grid_color':   '#E0E0E0',
    'font_size':    11,
    'title_size':   13,
}

def _setup_ax(ax, title, xlabel, ylabel):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=STYLE['title_size'],
                 fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=STYLE['font_size'])
    ax.set_ylabel(ylabel, fontsize=STYLE['font_size'])
    ax.set_facecolor(STYLE['bg_color'])
    ax.grid(True, color=STYLE['grid_color'], linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=STYLE['font_size'] - 1)


def _add_annotation(ax, text, x, y, color='black'):
    """Add a floating annotation box."""
    ax.annotate(
        text, xy=(x, y),
        xytext=(10, 10), textcoords='offset points',
        fontsize=STYLE['font_size'] - 2,
        color=color,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor=color, alpha=0.8)
    )


def plot_loss_curves(history: list, exp_name: str,
                     save_dir: str):
    """
    Plot 1: Train vs Validation loss over epochs.

    What to look for:
      - Both curves should decrease
      - Val loss should roughly track train loss
      - Large gap = overfitting (unlikely with only 77 train equations)
      - Val loss plateau = model has learned what it can
    """
    epochs    = [h['epoch']        for h in history]
    train_loss= [h['loss_total']   for h in history]
    val_loss  = [h['val_loss_lm']  for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    _setup_ax(ax,
              title  = f'[{exp_name}] Train vs Validation Loss',
              xlabel = 'Epoch',
              ylabel = 'Loss')

    ax.plot(epochs, train_loss, color=STYLE['train_color'],
            linewidth=2, label='Train Loss', zorder=3)
    ax.plot(epochs, val_loss,   color=STYLE['val_color'],
            linewidth=2, label='Val Loss',   zorder=3)

    # Shade the gap between train and val
    ax.fill_between(epochs, train_loss, val_loss,
                    alpha=0.08, color=STYLE['val_color'],
                    label='Train/Val Gap')

    # Annotate best validation loss
    best_epoch = epochs[np.argmin(val_loss)]
    best_val   = min(val_loss)
    ax.axvline(x=best_epoch, color=STYLE['val_color'],
               linestyle='--', alpha=0.5, linewidth=1)
    _add_annotation(ax,
                    f'Best val: {best_val:.4f}\n@ epoch {best_epoch}',
                    best_epoch, best_val,
                    color=STYLE['val_color'])

    # Annotate final values
    ax.annotate(f'Final: {train_loss[-1]:.4f}',
                xy=(epochs[-1], train_loss[-1]),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=STYLE['font_size'] - 2,
                color=STYLE['train_color'])

    ax.legend(fontsize=STYLE['font_size'])
    fig.tight_layout()

    path = os.path.join(save_dir, f'{exp_name}_loss_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_breakdown(history: list, exp_name: str,
                        save_dir: str):
    """
    Plot 2: Breakdown of loss components (LM, JEPA, SIGReg).

    What to look for:
      - LM loss should decrease steadily
      - JEPA loss should decrease — model is learning to predict
        equation embeddings from numerical context
      - SIGReg loss should stay low — if it spikes, collapse is starting
      - If JEPA loss stays HIGH while LM loss decreases, the model
        is learning syntax but not the data-to-symbol mapping

    Only meaningful for JEPA experiments (E1, E2, E3, E4, E5).
    For baselines (B1-B4) JEPA and SIGReg losses are zero.
    """
    epochs    = [h['epoch']      for h in history]
    lm_loss   = [h['loss_lm']    for h in history]
    jepa_loss = [h['loss_jepa']  for h in history]
    sig_loss  = [h['loss_sig']   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'[{exp_name}] Loss Component Breakdown',
                 fontsize=STYLE['title_size'] + 1,
                 fontweight='bold', y=1.02)

    # LM loss
    _setup_ax(axes[0],
              title  = 'Language Model Loss',
              xlabel = 'Epoch',
              ylabel = 'Cross-Entropy Loss')
    axes[0].plot(epochs, lm_loss, color=STYLE['lm_color'],
                 linewidth=2)
    axes[0].fill_between(epochs, lm_loss, alpha=0.15,
                          color=STYLE['lm_color'])
    axes[0].annotate(
        'Measures next-token prediction.\n'
        'Decreasing = learning equation syntax.',
        xy=(0.05, 0.92), xycoords='axes fraction',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # JEPA loss
    _setup_ax(axes[1],
              title  = 'JEPA Loss',
              xlabel = 'Epoch',
              ylabel = 'Cosine Distance (1 - similarity)')
    axes[1].plot(epochs, jepa_loss, color=STYLE['jepa_color'],
                 linewidth=2)
    axes[1].fill_between(epochs, jepa_loss, alpha=0.15,
                          color=STYLE['jepa_color'])
    axes[1].set_ylim(bottom=0, top=max(max(jepa_loss) * 1.2, 0.1))
    axes[1].annotate(
        'Measures prediction of equation\n'
        'embedding from numerical context.\n'
        'Decreasing = learning data→symbol.',
        xy=(0.05, 0.92), xycoords='axes fraction',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # SIGReg loss
    _setup_ax(axes[2],
              title  = 'SIGReg Loss',
              xlabel = 'Epoch',
              ylabel = 'Characteristic Function Deviation')
    axes[2].plot(epochs, sig_loss, color=STYLE['sig_color'],
                 linewidth=2)
    axes[2].fill_between(epochs, sig_loss, alpha=0.15,
                          color=STYLE['sig_color'])
    axes[2].axhline(y=0.05, color='red', linestyle='--',
                     alpha=0.5, linewidth=1,
                     label='Collapse warning threshold')
    axes[2].legend(fontsize=8)
    axes[2].annotate(
        'Measures deviation from isotropic\n'
        'Gaussian. Should stay low.\n'
        'Spike = collapse starting.',
        xy=(0.05, 0.92), xycoords='axes fraction',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    fig.tight_layout()
    path = os.path.join(save_dir, f'{exp_name}_loss_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learning_rate(history: list, exp_name: str,
                       save_dir: str):
    """
    Plot 3: Learning rate schedule over training.

    What to look for:
      - Smooth cosine decay from initial LR to eta_min
      - No sudden jumps (would indicate scheduler bug)
      - LR should reach eta_min at the final epoch
    """
    epochs = [h['epoch'] for h in history]
    lrs    = [h['lr']    for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    _setup_ax(ax,
              title  = f'[{exp_name}] Learning Rate Schedule',
              xlabel = 'Epoch',
              ylabel = 'Learning Rate')

    ax.plot(epochs, lrs, color=STYLE['lr_color'],
            linewidth=2, label='LR (cosine annealing)')
    ax.fill_between(epochs, lrs, alpha=0.15,
                    color=STYLE['lr_color'])

    # Annotate start and end
    ax.annotate(f'Start: {lrs[0]:.2e}',
                xy=(epochs[0], lrs[0]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=STYLE['font_size'] - 1,
                color=STYLE['lr_color'])
    ax.annotate(f'End: {lrs[-1]:.2e}',
                xy=(epochs[-1], lrs[-1]),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=STYLE['font_size'] - 1,
                color=STYLE['lr_color'])

    ax.set_yscale('log')
    ax.legend(fontsize=STYLE['font_size'])
    fig.tight_layout()

    path = os.path.join(save_dir, f'{exp_name}_lr_schedule.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_collapse_monitor(collapse_log: list, exp_name: str,
                          save_dir: str):
    """
    Plot 4: Embedding health monitoring.

    Logged during training by monitor_collapse() every N steps.

    What to look for:
      Variance:
        - Should stay close to 1.0
        - Dropping below 0.1 = collapse happening
        - If it drops to 0 = complete collapse, training is broken

      Mean cosine similarity:
        - Should stay close to 0.0
        - All embeddings pointing in random directions = healthy
        - Rising toward 1.0 = all embeddings becoming identical = collapse

    If you see collapse: increase lambda_sig in SIGReg.
    """
    if not collapse_log:
        print("  No collapse log data — skipping collapse plot")
        return

    steps    = [c['step']        for c in collapse_log]
    variance = [c['variance']    for c in collapse_log]
    cosine   = [c['mean_cosine'] for c in collapse_log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'[{exp_name}] Embedding Health Monitor',
                 fontsize=STYLE['title_size'] + 1,
                 fontweight='bold')

    # Variance plot
    _setup_ax(axes[0],
              title  = 'Embedding Variance',
              xlabel = 'Training Step',
              ylabel = 'Mean Variance Across Dimensions')
    axes[0].plot(steps, variance, color=STYLE['var_color'],
                 linewidth=2, label='Embedding variance')
    axes[0].axhline(y=1.0, color='green', linestyle='--',
                     alpha=0.6, linewidth=1,
                     label='Target (isotropic Gaussian)')
    axes[0].axhline(y=0.1, color='red', linestyle='--',
                     alpha=0.6, linewidth=1,
                     label='Collapse warning (< 0.1)')
    axes[0].fill_between(steps, 0, 0.1, alpha=0.05,
                          color='red', label='Collapse zone')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[0].annotate(
        'Variance measures spread of embeddings.\n'
        'Healthy range: 0.5 – 2.0\n'
        'Below 0.1: collapse is happening.',
        xy=(0.02, 0.05), xycoords='axes fraction',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Cosine similarity plot
    _setup_ax(axes[1],
              title  = 'Mean Pairwise Cosine Similarity',
              xlabel = 'Training Step',
              ylabel = 'Cosine Similarity (0=diverse, 1=collapsed)')
    axes[1].plot(steps, cosine, color=STYLE['cos_color'],
                 linewidth=2, label='Mean cosine similarity')
    axes[1].axhline(y=0.0, color='green', linestyle='--',
                     alpha=0.6, linewidth=1,
                     label='Target (orthogonal embeddings)')
    axes[1].axhline(y=0.9, color='red', linestyle='--',
                     alpha=0.6, linewidth=1,
                     label='Collapse warning (> 0.9)')
    axes[1].fill_between(steps, 0.9, 1.0, alpha=0.05,
                          color='red', label='Collapse zone')
    axes[1].set_ylim(-0.2, 1.1)
    axes[1].legend(fontsize=8)
    axes[1].annotate(
        'Cosine similarity measures how similar\n'
        'embedding directions are.\n'
        'Near 0: diverse (healthy)\n'
        'Near 1: all pointing same direction (collapse)',
        xy=(0.02, 0.05), xycoords='axes fraction',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    fig.tight_layout()
    path = os.path.join(save_dir, f'{exp_name}_collapse_monitor.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_card(history: list, exp_name: str,
                      exp_cfg: dict, save_dir: str):
    """
    Plot 5: Summary card — all key metrics in one figure.
    Good for the GSoC report and README.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F5F5F5')

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.35)

    epochs     = [h['epoch']       for h in history]
    train_loss = [h['loss_total']  for h in history]
    val_loss   = [h['val_loss_lm'] for h in history]
    lm_loss    = [h['loss_lm']     for h in history]
    jepa_loss  = [h['loss_jepa']   for h in history]
    sig_loss   = [h['loss_sig']    for h in history]
    lrs        = [h['lr']          for h in history]

    # ── Panel 1: Train/Val loss ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _setup_ax(ax1, 'Train vs Val Loss', 'Epoch', 'Loss')
    ax1.plot(epochs, train_loss, color=STYLE['train_color'],
             linewidth=2, label='Train')
    ax1.plot(epochs, val_loss,   color=STYLE['val_color'],
             linewidth=2, label='Val')
    ax1.legend(fontsize=9)

    # ── Panel 2: LM loss ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _setup_ax(ax2, 'LM Loss', 'Epoch', 'Cross-Entropy')
    ax2.plot(epochs, lm_loss, color=STYLE['lm_color'], linewidth=2)

    # ── Panel 3: JEPA loss ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _setup_ax(ax3, 'JEPA Loss', 'Epoch', 'Cosine Distance')
    ax3.plot(epochs, jepa_loss, color=STYLE['jepa_color'], linewidth=2)

    # ── Panel 4: SIGReg loss ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _setup_ax(ax4, 'SIGReg Loss', 'Epoch', 'CF Deviation')
    ax4.plot(epochs, sig_loss, color=STYLE['sig_color'], linewidth=2)
    ax4.axhline(y=0.05, color='red', linestyle='--',
                 alpha=0.5, linewidth=1)

    # ── Panel 5: Learning rate ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _setup_ax(ax5, 'Learning Rate', 'Epoch', 'LR')
    ax5.plot(epochs, lrs, color=STYLE['lr_color'], linewidth=2)
    ax5.set_yscale('log')

    # ── Panel 6: Summary stats text box ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    best_val   = min(val_loss)
    best_epoch = epochs[np.argmin(val_loss)]
    final_lm   = lm_loss[-1]

    summary = (
        f"EXPERIMENT: {exp_name}\n"
        f"{'─'*28}\n"
        f"JEPA enabled : {exp_cfg['use_jepa']}\n"
        f"λ_jepa       : {exp_cfg['lambda_lejepa']}\n"
        f"{'─'*28}\n"
        f"Total epochs : {len(history)}\n"
        f"Best val loss: {best_val:.4f}\n"
        f"Best epoch   : {best_epoch}\n"
        f"Final LM loss: {final_lm:.4f}\n"
        f"Final JEPA   : {jepa_loss[-1]:.4f}\n"
        f"Final SIGReg : {sig_loss[-1]:.4f}\n"
        f"{'─'*28}\n"
        f"Min LR       : {min(lrs):.2e}\n"
        f"Max LR       : {max(lrs):.2e}\n"
    )

    ax6.text(0.05, 0.95, summary,
             transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor='#BDBDBD', alpha=0.9))

    # ── Title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'LM-JEPA Training Summary — {exp_name}',
        fontsize=16, fontweight='bold', y=1.01
    )

    path = os.path.join(save_dir, f'{exp_name}_summary_card.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_plots(history: list, exp_name: str,
                       exp_cfg: dict,
                       collapse_log: list = None,
                       save_dir: str = 'results/plots'):
    """
    Generate all plots for one experiment.
    Call this at the end of training.

    Args:
        history:      list of metric dicts from training loop
        exp_name:     e.g. 'B1', 'E2'
        exp_cfg:      experiment config dict
        collapse_log: list of collapse monitor dicts (optional)
        save_dir:     directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating plots for {exp_name}...")

    plot_loss_curves(history, exp_name, save_dir)
    plot_loss_breakdown(history, exp_name, save_dir)
    plot_learning_rate(history, exp_name, save_dir)
    plot_summary_card(history, exp_name, exp_cfg, save_dir)

    if collapse_log:
        plot_collapse_monitor(collapse_log, exp_name, save_dir)

    print(f"All plots saved to {save_dir}/")