import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# final GM scores used in paper
data_g_mean = {
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
data_auc_score = {
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

    # initialize data and convert them into DataFrame format
    _data = data_auc_score
    df = pd.DataFrame(_data)

    # Convert to long format for seaborn
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Extract the mean and standard deviation
    average_df = df[["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(_data["STD"])  # Convert to NumPy array for standard deviation

    # Set theme
    sns.set_theme(style="darkgrid")

    # Choose a color palette (e.g., "Set1" which will have consistent colors for datasets)
    palette = sns.color_palette("Set1", n_colors=1)  # We need only one color for the mean

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot each dataset individually in black
    for idx, dataset in enumerate(df.columns[1:-2]):  # Skip the IR and Average columns
        sns.lineplot(
            data=df, x="IR", y=dataset,
            color="black", linewidth=1.5, alpha=0.2
        )

    # Plot the mean AUC trend using a different color
    sns.lineplot(
        data=average_df, x="IR", y="AUC",
        color=palette[0], linewidth=3, label="Average AUC score"
    )

    # Convert categorical x-axis labels to numerical indices for fill_between
    x_values = np.arange(len(average_df["IR"]))

    # Add standard deviation shading around the mean line
    low_boundary = (average_df["AUC"] - std_dev).clip(lower=0)
    plt.fill_between(
        x_values,
        low_boundary,
        average_df["AUC"] + std_dev,
        color="gray", alpha=0.2, label="Standard deviation"
    )

    # Adjust x-ticks
    plt.xticks(x_values, average_df["IR"])  # Set labels correctly

    # Labels and title
    plt.xlabel("Imbalance ratio", fontsize=20)
    plt.ylabel("AUC score", fontsize=20)
    plt.xticks(fontsize=20)  # X-axis tick labels
    plt.yticks(fontsize=20)  # Y-axis tick labels

    # Show legend
    plt.legend(fontsize=20)

    # Show the plot
    plt.savefig("./ir_synthetic_graph_analysis_auc_scores.png", dpi=300, bbox_inches='tight')
    plt.show()
