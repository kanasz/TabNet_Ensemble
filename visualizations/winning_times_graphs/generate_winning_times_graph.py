import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Data
# -------------------------
data = {
    'Method': [
        'W-SVM',
        'SMOTE+SVM',
        'ADASYN+SVM',
        'SMOTE+XGBoost',
        'AdaCost',
        'SelfPE',
        'SMOTE+TabNet',
        'GA-HESO'
    ],
    'AUC': [3, 2, 0, 2, 4, 3, 2, 15],
    'GM':  [3, 2, 0, 2, 4, 3, 2, 15]
}


if __name__ == "__main__":
    df = pd.DataFrame(data)

    # -------------------------
    # Style
    # -------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 22,
        "xtick.labelsize": 15,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.linewidth": 1.4,

        # Better PDF fonts
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.set_facecolor("white")

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Bars
    # -------------------------
    x = np.arange(len(df))
    width = 0.36

    ax.bar(
        x - width/2,
        df["AUC"],
        width,
        label="AUC",
        color="#0057B8",
        edgecolor="black",
        linewidth=0.8
    )

    ax.bar(
        x + width/2,
        df["GM"],
        width,
        label="GM",
        color="#F28E2B",
        edgecolor="black",
        linewidth=0.8
    )

    # -------------------------
    # Labels
    # -------------------------
    ax.set_ylabel("Number of Wins")
    ax.set_xlabel("")

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Method"],
        rotation=40,
        ha="right"
    )

    ax.set_yticks(range(0, 17, 2))

    ax.set_ylim(0, 16)

    # Legend
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black"
    )

    plt.tight_layout()

    # -------------------------
    # Output
    # -------------------------
    output_dir = Path("")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "winning_times_rate_v02.pdf",
        bbox_inches="tight"
    )

    plt.show()
