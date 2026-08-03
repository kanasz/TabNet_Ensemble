import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# Output directory
output_dir = Path("")
output_dir.mkdir(parents=True, exist_ok=True)

FILE_NAME = "imbalance_severity_synthetic_auc_v02.pdf"

DATA_G_MEAN = {
    'IR': ['1:2', '1:5', '1:10', '1:20', '1:50'],
    'synthetic_dataset_1': [0.921226863, 0.8901344385, 0.744159836, 0.6766647619, 0],
    'synthetic_dataset_3': [0.7538936175, 0.7643666046, 0.7145288841, 0.6844163332, 0.6861289854],
    'synthetic_dataset_4': [0.655249808, 0.717399458, 0.6594950023, 0.6072789153, 0.6306215863],
    'synthetic_dataset_5': [0.7750999741, 0.8153043762, 0.7028631348, 0.6921540815, 0],
    'synthetic_dataset_6': [0.815148836, 0.8091105673, 0.8736537029, 0.7813820399, 0],
    'synthetic_dataset_7': [0.7732558958, 0.8125129655, 0.6081326628, 0.3713440868, 0],
    'synthetic_dataset_8': [0.8632880151, 0.8315107721, 0.67391173, 0.5097094777, 0],
    'Average': [0.7938804299, 0.8057627403, 0.7109635647, 0.6175642423, 0.1881072245],
    'STD': [0.06396418116, 0.03940293277, 0.08264888364, 0.134618559, 0.3110002793]
}

# final AUC scores used in paper
DATA_AUC_SCORE = {
    'IR': ['1:2', '1:5', '1:10', '1:20', '1:50'],
    'synthetic_dataset_1': [0.9223596257, 0.8920634921, 0.7694202899, 0.7270833333, 0.5],
    'synthetic_dataset_3': [0.7585840849, 0.7822099322, 0.7253968254, 0.7011184211, 0.7044011544],
    'synthetic_dataset_4': [0.6579415755, 0.7255695049, 0.6666666667, 0.6203837719, 0.6421665636],
    'synthetic_dataset_5': [0.7775, 0.828, 0.7251515152, 0.7186630369, 0.5],
    'synthetic_dataset_6': [0.8175, 0.83, 0.8774074074, 0.7950998185, 0.5],
    'synthetic_dataset_7': [0.7775, 0.814, 0.6503030303, 0.6149122807, 0.5],
    'synthetic_dataset_8': [0.865, 0.834, 0.6792592593, 0.6666666667, 0.5],
    'Average': [0.7966264694, 0.8151204184, 0.7276578563, 0.6919896184, 0.549509674],
    'STD': [0.06359324734, 0.03873348604, 0.07553507277, 0.06186025431, 0.08369720089]
}


if __name__ == "__main__":
    df = pd.DataFrame(DATA_AUC_SCORE)

    average = df["Average"].values
    std = df["STD"].values
    x = np.arange(len(df["IR"]))

    # -------------------------
    # Figure style
    # -------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # White background
    ax.set_facecolor("white")

    # Light grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color="#d9d9d9", alpha=0.7)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Individual datasets
    # -------------------------
    for dataset in df.columns[1:-2]:
        ax.plot(
            x,
            df[dataset],
            color="gray",
            linewidth=2,
            alpha=0.45,
            zorder=1
        )

    # -------------------------
    # Standard deviation band
    # -------------------------
    ax.fill_between(
        x,
        np.maximum(0, average - std),
        average + std,
        color="#5B9BD5",
        alpha=0.25,
        zorder=2
    )

    # -------------------------
    # Average line
    # -------------------------
    ax.plot(
        x,
        average,
        color="#0057B8",
        linewidth=4,
        marker='o',
        markersize=14,
        markeredgecolor="#0057B8",
        zorder=3
    )

    # -------------------------
    # Labels
    # -------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(df["IR"])

    ax.set_xlabel("Imbalance Ratio")
    ax.set_ylabel("AUC")

    # Optional
    ax.set_ylim(0.45, 0.95)

    # -------------------------
    # Custom legend
    # -------------------------
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#0057B8",
            marker='o',
            linewidth=4,
            markersize=12,
            label="Avg. AUC"
        ),
        Patch(
            facecolor="#5B9BD5",
            edgecolor="#5B9BD5",
            alpha=0.25,
            label="Standard Deviation"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        fancybox=True,
        framealpha=1,
        borderpad=0.8
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / FILE_NAME,
        bbox_inches="tight"
    )

    # plt.savefig(
    #     "auc_scores.png",
    #     dpi=600,
    #    bbox_inches="tight"
    # )

    plt.show()
