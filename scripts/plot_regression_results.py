import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_history(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    epochs = payload.get("epochs", [])
    if not epochs:
        raise ValueError(f"No epoch history found in {path}")
    return epochs


def plot_losses(epochs: list[dict], output_path: Path) -> None:
    epoch_ids = [entry["epoch"] for entry in epochs]
    train_losses = [entry["train_loss"] for entry in epochs]
    val_losses = [entry["val_loss"] for entry in epochs]

    best_idx = min(range(len(epochs)), key=lambda idx: epochs[idx]["val_loss"])
    best_epoch = epoch_ids[best_idx]
    best_val_loss = val_losses[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_ids, train_losses, marker="o", linewidth=2, label="Train loss")
    plt.plot(epoch_ids, val_losses, marker="o", linewidth=2, linestyle="--", label="Val loss")
    plt.scatter([best_epoch], [best_val_loss], color="red", zorder=5, label=f"Best val epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(epoch_ids)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_validation_scatter(predictions_df: pd.DataFrame, output_path: Path) -> None:
    if predictions_df.empty:
        raise ValueError("Validation predictions file is empty")

    targets = predictions_df["target"]
    predictions = predictions_df["prediction"]
    residuals = predictions - targets
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    r2 = 1 - (residuals ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    lower = min(targets.min(), predictions.min())
    upper = max(targets.max(), predictions.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, alpha=0.65, s=20)
    plt.plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1.5, label="Ideal")
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title(f"Validation Predictions  (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_age_distribution(predictions_df: pd.DataFrame, output_path: Path) -> None:
    targets = predictions_df["target"]
    predictions = predictions_df["prediction"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(min(targets.min(), predictions.min()), max(targets.max(), predictions.max()), 30)
    axes[0].hist(targets, bins=bins, alpha=0.6, label="True age", color="steelblue")
    axes[0].hist(predictions, bins=bins, alpha=0.6, label="Predicted age", color="coral")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].set_title("True vs Predicted Age Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(predictions - targets, bins=25, color="mediumpurple", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (predicted − true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_by_age_bin(predictions_df: pd.DataFrame, output_path: Path, n_bins: int = 8) -> None:
    targets = predictions_df["target"]
    predictions = predictions_df["prediction"]
    residuals = np.abs(predictions - targets)

    bins = pd.cut(targets, bins=n_bins)
    bin_labels = [str(interval) for interval in sorted(bins.cat.categories)]
    mae_per_bin = predictions_df.assign(residual=residuals, bin=bins).groupby("bin", observed=True)["residual"].mean()
    count_per_bin = predictions_df.assign(bin=bins).groupby("bin", observed=True).size()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bin_labels))
    bars = ax1.bar(x, mae_per_bin.values, color="steelblue", alpha=0.8, label="MAE")
    ax1.set_xlabel("True age bin")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax1.set_title("Prediction Error by Age Bin")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, count_per_bin.values, color="coral", marker="o", linewidth=2, label="Count")
    ax2.set_ylabel("Sample count", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_val_metrics(epochs: list[dict], output_path: Path) -> None:
    epoch_ids = [e["epoch"] for e in epochs]
    rmse = [e["val_rmse"] for e in epochs]
    mae = [e["val_mae"] for e in epochs]
    r2 = [e["val_r2"] for e in epochs]

    best_idx = min(range(len(epochs)), key=lambda i: epochs[i]["val_rmse"])
    best_epoch = epoch_ids[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, label, color in zip(
        axes,
        [rmse, mae, r2],
        ["Val RMSE", "Val MAE", "Val R²"],
        ["steelblue", "coral", "mediumseagreen"],
    ):
        ax.plot(epoch_ids, values, marker="o", linewidth=2, color=color)
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1, label=f"Best epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves and validation predictions for a regression run")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Directory containing history.json and val_predictions.csv",
    )
    args = parser.parse_args()

    history_path = args.run_dir / "history.json"
    predictions_path = args.run_dir / "val_predictions.csv"

    epochs = load_history(history_path)
    predictions_df = pd.read_csv(predictions_path)

    plots = [
        ("loss_curve.png",            lambda p: plot_losses(epochs, p)),
        ("val_predictions_scatter.png", lambda p: plot_validation_scatter(predictions_df, p)),
        ("age_distribution.png",       lambda p: plot_age_distribution(predictions_df, p)),
        ("error_by_age_bin.png",        lambda p: plot_error_by_age_bin(predictions_df, p)),
        ("val_metrics.png",             lambda p: plot_val_metrics(epochs, p)),
    ]

    for filename, plot_fn in plots:
        out = args.run_dir / filename
        plot_fn(out)
        print(f"Saved {filename} to {out}")


if __name__ == "__main__":
    main()
