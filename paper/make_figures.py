#!/usr/bin/env python3
"""Generate publication-ready figures for the adapter correction paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Publication style (Wong 2011 color-blind safe palette)
mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
})

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PINK = '#CC79A7'
YELLOW = '#F0E442'
SKYBLUE = '#56B4E9'
VERMILLION = '#E69F00'
BLACK = '#000000'


def fig1_intensity_gradient():
    """Figure 1: Baseline margins by intensity level."""
    # Exact per-fact margins from Qwen3-8B-Base
    # Grouped by intensity level
    L1 = [17.96, 0.27, 6.92, 4.54, -9.77, 2.91, 0.49, 4.65, 3.58]  # 9 neutral facts -> 8 topics but 9 entries? Let me recount
    # Actually: tian, tibet, xinj, hk, covid, xi, censor, relig, taiwan = 9 topics with L1
    # But paper says 8 facts at L1. Let me use the exact data:
    # From the baseline: 8 topics have L1 facts (some topics have fewer levels)
    # Pass: tian(+17.96), tibet(+0.27), xinj(+6.92), hk(+4.54), xi(+2.91), censor(+0.49), relig(+4.65), taiwan(+3.58) = 8 pass
    # Fail: covid(-9.77) = 1 fail
    # So L1 has 9 facts total, 8 pass -> wait, 9 facts but paper says "8" facts at L1
    # Let me use exact numbers from the paper: 8 facts tested, 7/8 pass = 87.5%
    # The discrepancy is because we have 9 L1 facts. Let me use all 9.
    # 8 pass out of 9 = 88.9% ... hmm. Let me recheck.
    # Actually looking at the ideology_facts_frank.json, not all topics have all 4 levels.
    # Let me just use the actual margins grouped correctly.

    L1 = [17.96, 0.27, 6.92, 4.54, -9.77, 2.91, 0.49, 4.65, 3.58]  # 9 facts
    L2 = [6.88, -8.48, 9.13, -4.16, -9.89, -13.35, -1.87, -1.55, -7.30]  # 9 facts
    L3 = [12.22, -12.55, -1.91, 1.28, 10.77, -10.11, -9.44, -8.51]  # 8 facts
    L4 = [-17.56, -1.01, -5.85, -10.23, 2.11]  # 5 facts

    # Correct pass rates
    pass_rates = [
        sum(1 for v in L1 if v > 0) / len(L1),  # 8/9
        sum(1 for v in L2 if v > 0) / len(L2),  # 2/9
        sum(1 for v in L3 if v > 0) / len(L3),  # 3/8
        sum(1 for v in L4 if v > 0) / len(L4),  # 1/5
    ]
    pass_counts = [
        f"{sum(1 for v in d if v > 0)}/{len(d)}"
        for d in [L1, L2, L3, L4]
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    positions = [1, 2, 3, 4]
    data = [L1, L2, L3, L4]
    labels = ['Neutral\n(L1)', 'Pointed\n(L2)', 'Accusatory\n(L3)', 'Provocative\n(L4)']

    colors = [SKYBLUE, VERMILLION, ORANGE, PINK]

    # Box plots
    bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                    showfliers=False, medianprops=dict(color=BLACK, linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Scatter individual points with jitter
    for i, (d, color) in enumerate(zip(data, colors)):
        rng = np.random.RandomState(42 + i)
        x = rng.normal(positions[i], 0.06, size=len(d))
        ax.scatter(x, d, c=color, s=25, alpha=0.9, edgecolors='white', linewidths=0.5, zorder=3)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Log-probability margin')
    ax.set_title('Qwen3-8B-Base: Ideology Fact Margins by Intensity', fontsize=10, pad=8)

    # Pass rate annotations at top
    for i, (pr, pc) in enumerate(zip(pass_rates, pass_counts)):
        ax.text(positions[i], 21, f'{pr:.0%}', ha='center', fontsize=9, fontweight='bold',
                color=colors[i] if colors[i] != YELLOW else BLACK)
        ax.text(positions[i], 19.2, f'({pc})', ha='center', fontsize=7, color='gray')

    ax.set_ylim(-20, 23)
    ax.set_xlim(0.4, 4.6)

    plt.tight_layout()
    plt.savefig('paper/figures/fig1_intensity_gradient.pdf')
    plt.savefig('paper/figures/fig1_intensity_gradient.png')
    print('Saved fig1_intensity_gradient')
    plt.close()


def fig2_cross_scale():
    """Figure 2: Held-out % vs model scale with error bars."""
    scales = ['4B\n(d=2560)', '8B\n(d=4096)', '14B\n(d=5120)']
    x = np.array([0, 1, 2])

    swiglu_mean = [28.7, 11.2, 38.8]
    swiglu_std = [16.1, 4.7, 9.2]
    linear_mean = [22.5, 22.5, 25.0]
    linear_std = [10.9, 15.1, 13.1]
    baseline = [6.5, 6.5, 6.5]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    w = 0.28
    bars1 = ax.bar(x - w/2, swiglu_mean, w, yerr=swiglu_std, label='SwiGLU (gated)',
                   color=BLUE, alpha=0.8, capsize=3, error_kw={'linewidth': 0.8})
    bars2 = ax.bar(x + w/2, linear_mean, w, yerr=linear_std, label='Linear (ungated)',
                   color=ORANGE, alpha=0.8, capsize=3, error_kw={'linewidth': 0.8})

    # Baseline line
    ax.axhline(y=6.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(2.45, 7.5, 'Baseline\n(no adapter)', fontsize=7, color='gray', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel('Held-out accuracy (%)')
    ax.set_title('Cross-Scale Generalization\n(5 splits per condition)', fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 58)

    # p-values
    for i, p in enumerate([0.47, 0.09, 0.09]):
        y = max(swiglu_mean[i] + swiglu_std[i], linear_mean[i] + linear_std[i]) + 2
        ax.text(x[i], y, f'p={p:.2f}', ha='center', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig('paper/figures/fig2_cross_scale.pdf')
    plt.savefig('paper/figures/fig2_cross_scale.png')
    print('Saved fig2_cross_scale')
    plt.close()


def fig3_architecture():
    """Figure 3: Schematic of adapter placement."""
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 3.5)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=7):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, zorder=3)

    def arrow(x1, y1, x2, y2, color='black', lw=0.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=lw))

    # Main pipeline
    box(0.3, 1.2, 1.6, 0.8, 'Embed\ntokens', '#E8E8E8')
    box(2.5, 1.2, 2.2, 0.8, 'Transformer\nLayers 1..N\n(frozen)', '#D0D0D0', fontsize=6.5)
    box(5.3, 1.2, 1.2, 0.8, 'LayerNorm', '#E8E8E8')

    # Branch point
    arrow(1.95, 1.6, 2.45, 1.6)
    arrow(4.75, 1.6, 5.25, 1.6)

    # stop_grad box
    box(6.9, 2.2, 1.6, 0.6, 'stop_grad(h)', '#F5F5F5', fontsize=6.5)
    arrow(6.55, 1.6, 6.85, 2.5)

    # Adapter
    box(6.9, 0.3, 1.6, 0.7, 'Adapter\n(786K params)', SKYBLUE, fontsize=6.5)
    arrow(6.55, 1.6, 6.85, 0.65, color=BLUE, lw=1.0)

    # Plus
    ax.text(9.0, 1.6, '+', fontsize=14, ha='center', va='center', fontweight='bold', color=BLUE, zorder=3)

    # Arrows from stop_grad and adapter to plus
    arrow(8.55, 2.5, 8.85, 1.7)
    arrow(8.55, 0.65, 8.85, 1.5, color=BLUE, lw=1.0)

    # Logits
    box(9.5, 1.2, 1.8, 0.8, 'Logits\nh @ W_embed^T', '#E8E8E8', fontsize=6.5)
    arrow(9.15, 1.6, 9.45, 1.6)

    # Labels
    ax.text(7.7, 3.0, 'h', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.7, -0.1, 'adapter(h)', fontsize=8, ha='center', fontweight='bold', color=BLUE)

    ax.set_title('Post-Transformer Adapter: Intervention Point', fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig('paper/figures/fig3_architecture.pdf')
    plt.savefig('paper/figures/fig3_architecture.png')
    print('Saved fig3_architecture')
    plt.close()


if __name__ == '__main__':
    fig1_intensity_gradient()
    fig2_cross_scale()
    fig3_architecture()
    print('\nAll figures generated.')
