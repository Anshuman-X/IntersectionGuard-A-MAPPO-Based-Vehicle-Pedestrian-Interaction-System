"""
rl/plot_results.py
==================
Publication-quality result visualisations for the MAPPO training run.

Generates 5 figures from results/training_log.csv:

  1. learning_curve.png      — Episode total reward over training
  2. ttc_improvement.png     — Average TTC per episode (safety improvement)
  3. pet_improvement.png     — Average PET per episode
  4. conflict_reduction.png  — Total conflicts per episode
  5. losses.png              — Actor + Critic loss curves

All figures use a consistent, publication-ready dark style with:
  - Smoothed trend lines (exponential moving average)
  - Error bands (rolling std)
  - Proper axis labels, titles, and legends
  - 300 DPI output suitable for paper submission

Usage
-----
    python -m rl.plot_results
    python -m rl.plot_results --log results/training_log.csv --out results/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")       # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Style configuration ───────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d2e",
    "axes.edgecolor":    "#3a3d52",
    "axes.labelcolor":   "#e0e0f0",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#a0a0c0",
    "ytick.color":       "#a0a0c0",
    "grid.color":        "#2a2d3e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#e0e0f0",
    "legend.facecolor":  "#1a1d2e",
    "legend.edgecolor":  "#3a3d52",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
}

PALETTE = {
    "reward":    "#4fc3f7",   # Sky blue
    "ttc":       "#81c784",   # Soft green (good = high TTC)
    "pet":       "#aed6f1",   # Light blue (good = high PET)
    "conflict":  "#ef9a9a",   # Soft red (bad = high conflicts)
    "actor":     "#ffb74d",   # Amber (actor losses)
    "critic":    "#ce93d8",   # Lavender (critic loss)
    "band":      0.15,         # Alpha for shaded bands
}


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Exponential moving average smoothing for learning curves.
    Makes trend visible through episode-to-episode variance.
    """
    alpha  = 2.0 / (window + 1)
    ema    = np.zeros_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


def rolling_std(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute rolling standard deviation for error bands."""
    series = pd.Series(values)
    return series.rolling(window, min_periods=1).std().fillna(0).values


def apply_style():
    """Apply dark publication style to all subsequent plots."""
    plt.rcParams.update(STYLE)


def save_fig(fig: plt.Figure, path: str):
    """Save figure with tight layout at 300 DPI."""
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_learning_curve(df: pd.DataFrame, out_dir: str):
    """
    Plot 1: Episode total reward vs training episode.

    An upward trend confirms MAPPO is learning a safer policy.
    The shaded band shows reward variance — narrowing over time
    indicates more consistent agent behaviour.
    """
    apply_style()
    episodes = df["episode"].values
    rewards  = df["total_reward"].values
    smoothed = smooth(rewards)
    std      = rolling_std(rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(episodes, smoothed - std, smoothed + std,
                    color=PALETTE["reward"], alpha=PALETTE["band"], label="_nolegend_")
    ax.plot(episodes, rewards,  color=PALETTE["reward"], alpha=0.3,  linewidth=0.8)
    ax.plot(episodes, smoothed, color=PALETTE["reward"], linewidth=2.5, label="EMA Reward")
    ax.axhline(y=0, color="#ffffff", linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("MAPPO Learning Curve — IntersectionGuard-v0")
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    fig.tight_layout()

    save_fig(fig, os.path.join(out_dir, "learning_curve.png"))


def plot_ttc_improvement(df: pd.DataFrame, out_dir: str):
    """
    Plot 2: Average TTC per episode.

    TTC (Time-to-Collision) is the primary safety metric in PVCA.
    A rising TTC curve means agents learn to maintain safer headways
    between vehicles and pedestrians over training episodes.
    """
    apply_style()
    episodes = df["episode"].values
    ttc      = df["avg_ttc"].values
    smoothed = smooth(ttc)
    std      = rolling_std(ttc)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(episodes, smoothed - std, smoothed + std,
                    color=PALETTE["ttc"], alpha=PALETTE["band"])
    ax.plot(episodes, ttc,      color=PALETTE["ttc"], alpha=0.3,  linewidth=0.8)
    ax.plot(episodes, smoothed, color=PALETTE["ttc"], linewidth=2.5, label="Avg TTC (EMA)")
    ax.axhline(y=3.0, color="#ef9a9a", linewidth=1.2, linestyle="--",
               label="Safety Threshold (3.0s)", alpha=0.8)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average TTC (seconds)")
    ax.set_title("TTC Improvement Over Training — Pedestrian Safety")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    fig.tight_layout()

    save_fig(fig, os.path.join(out_dir, "ttc_improvement.png"))


def plot_pet_improvement(df: pd.DataFrame, out_dir: str):
    """
    Plot 3: Average PET per episode.

    PET (Post-Encroachment Time) is the temporal gap between a vehicle
    exiting a conflict zone and a pedestrian entering it. Higher PET
    indicates better temporal separation and safer crossing conditions.
    """
    apply_style()
    episodes = df["episode"].values
    pet      = df["avg_pet"].values
    smoothed = smooth(pet)
    std      = rolling_std(pet)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(episodes, smoothed - std, smoothed + std,
                    color=PALETTE["pet"], alpha=PALETTE["band"])
    ax.plot(episodes, pet,      color=PALETTE["pet"], alpha=0.3,  linewidth=0.8)
    ax.plot(episodes, smoothed, color=PALETTE["pet"], linewidth=2.5, label="Avg PET (EMA)")
    ax.axhline(y=5.0, color="#ffb74d", linewidth=1.2, linestyle="--",
               label="PET Threshold (5.0s)", alpha=0.8)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average PET (seconds)")
    ax.set_title("PET Improvement Over Training — Temporal Conflict Gap")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    fig.tight_layout()

    save_fig(fig, os.path.join(out_dir, "pet_improvement.png"))


def plot_conflict_reduction(df: pd.DataFrame, out_dir: str):
    """
    Plot 4: Total conflicts per episode.

    A downward trend in conflicts is the clearest evidence that MAPPO
    is learning a safer intersection management policy. This is the key
    figure for the paper's results section.
    """
    apply_style()
    episodes  = df["episode"].values
    conflicts = df["total_conflicts"].values
    smoothed  = smooth(conflicts)
    std       = rolling_std(conflicts)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(episodes, np.maximum(smoothed - std, 0), smoothed + std,
                    color=PALETTE["conflict"], alpha=PALETTE["band"])
    ax.plot(episodes, conflicts, color=PALETTE["conflict"], alpha=0.3, linewidth=0.8)
    ax.plot(episodes, smoothed,  color=PALETTE["conflict"], linewidth=2.5, label="Total Conflicts (EMA)")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Conflicts per Episode")
    ax.set_title("Conflict Reduction Over Training — PVCA Safety Metric")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    save_fig(fig, os.path.join(out_dir, "conflict_reduction.png"))


def plot_losses(df: pd.DataFrame, out_dir: str):
    """
    Plot 5: Actor and Critic loss curves.

    Converging losses confirm that the neural networks are learning
    stable representations. Critic loss (value function MSE) should
    fall as V(s) becomes a better predictor of returns.
    """
    apply_style()
    episodes = df["episode"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Critic loss ─────────────────────────────────────────────────
    critic_loss = df["critic_loss"].values
    critic_sm   = smooth(critic_loss)
    ax1.plot(episodes, critic_loss, color=PALETTE["critic"], alpha=0.3, linewidth=0.8)
    ax1.plot(episodes, critic_sm,   color=PALETTE["critic"], linewidth=2.5, label="Critic Loss")
    ax1.set_ylabel("Critic Loss (MSE)")
    ax1.set_title("Neural Network Training Losses")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # ── Actor losses (one per agent) ────────────────────────────────
    actor_cols = {
        "actor_loss_west":  ("West Actor",  "#ffb74d"),
        "actor_loss_east":  ("East Actor",  "#4fc3f7"),
        "actor_loss_north": ("North Actor", "#81c784"),
    }
    for col, (label, color) in actor_cols.items():
        if col in df.columns:
            vals = df[col].values
            sm   = smooth(vals)
            ax2.plot(episodes, vals, color=color, alpha=0.3, linewidth=0.8)
            ax2.plot(episodes, sm,   color=color, linewidth=2.0, label=label)

    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Actor Loss (PPO Clip)")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "losses.png"))


def print_summary(df: pd.DataFrame):
    """Print a terminal summary of training results."""
    n = len(df)
    early = df.head(min(20, n // 4))
    late  = df.tail(min(20, n // 4))

    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    print(f"  Total episodes         : {n}")
    print(f"  Avg reward  (early)    : {early['total_reward'].mean():.2f}")
    print(f"  Avg reward  (late)     : {late['total_reward'].mean():.2f}")
    print(f"  Avg TTC     (early)    : {early['avg_ttc'].mean():.3f}s")
    print(f"  Avg TTC     (late)     : {late['avg_ttc'].mean():.3f}s")
    print(f"  Avg PET     (early)    : {early['avg_pet'].mean():.3f}s")
    print(f"  Avg PET     (late)     : {late['avg_pet'].mean():.3f}s")
    print(f"  Avg conflicts (early)  : {early['total_conflicts'].mean():.1f}")
    print(f"  Avg conflicts (late)   : {late['total_conflicts'].mean():.1f}")
    ttc_imp = late['avg_ttc'].mean() - early['avg_ttc'].mean()
    con_red = early['total_conflicts'].mean() - late['total_conflicts'].mean()
    print(f"\n  TTC improvement        : {ttc_imp:+.3f}s")
    print(f"  Conflict reduction     : {con_red:+.1f} per episode")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot MAPPO training results")
    parser.add_argument("--log", type=str, default="results/training_log.csv",
                        help="Path to training_log.csv")
    parser.add_argument("--out", type=str, default="results/",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"[Error] Log file not found: {args.log}")
        print("  Run training first:  python -m rl.train_mappo")
        return

    df = pd.read_csv(args.log)
    if len(df) < 2:
        print(f"[Warning] Only {len(df)} episodes logged — need more data for meaningful plots.")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"\nGenerating publication figures from {args.log}...\n")

    plot_learning_curve(df, args.out)
    plot_ttc_improvement(df, args.out)
    plot_pet_improvement(df, args.out)
    plot_conflict_reduction(df, args.out)
    plot_losses(df, args.out)
    print_summary(df)

    print(f"All figures saved to {args.out}")


if __name__ == "__main__":
    main()
