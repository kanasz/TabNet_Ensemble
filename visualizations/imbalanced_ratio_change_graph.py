import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = {
    'IR': ['1:2', '1:5', '1:10', '1:20', '1:50'],
    'synthetic_dataset_1': [0.921226863,	0.8901344385, 0.744159836, 0.6766647619, 0],
    'synthetic_dataset_2': [0.6063909389, 0.6277958747, 0.6705910909, 0.7298313396, 0],
    'synthetic_dataset_3': [0.7538936175, 0.7643666046, 0.7145288841, 0.6844163332, 0.6861289854],
    'synthetic_dataset_4': [0.655249808, 0.717399458, 0.6594950023, 0.6072789153, 0.6306215863],
    'synthetic_dataset_5': [0.7750999741, 0.8153043762, 0.7028631348, 0.6921540815, 0],
    'synthetic_dataset_6': [0.815148836, 0.8091105673, 0.8736537029, 0.7813820399, 0],
    'synthetic_dataset_7': [0.7732558958, 0.8125129655, 0.6081326628, 0.3713440868, 0],
    'synthetic_dataset_8': [0.8632880151, 0.8315107721, 0.67391173, 0.5097094777, 0],
    'Average': [0.7704442436,	0.7835168821, 0.7059170055, 0.6315976295, 0.1645938215],
    'STD': [0.08303466649, 0.06800843851, 0.07748300637, 0.13172931, 0.2979093928]
}

"""
'Average score': [0.7704442436,	0.7835168821, 0.7059170055, 0.6315976295, 0.1645938215],
'STD': [0.08303466649, 0.06800843851, 0.07748300637, 0.13172931, 0.2979093928]
"""
if __name__ == "__main__":

    """
    # Convert data into a long format DataFrame
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Separate the average and standard deviation
    average_df = df[df["Average"].notna()][["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(data["STD"])  # Convert to NumPy array for broadcasting

    # Set theme
    sns.set_theme(style="darkgrid")

    # Create the faceted plot
    g = sns.relplot(
        data=df_melted[df_melted["Dataset"].str.startswith("dataset")],  # Exclude average for individual plots
        x="IR", y="AUC", col="Dataset", hue="Dataset",
        kind="line", palette="crest", linewidth=2.5, zorder=5,
        col_wrap=4, height=2.5, aspect=1.2, legend=False
    )

    # Iterate over each subplot to customize further
    for dataset, ax in g.axes_dict.items():
        # Add the dataset title within each subplot
        ax.text(0.75, 0.85, dataset, transform=ax.transAxes, fontweight="bold")

        # Plot the mean trend in the background
        sns.lineplot(
            data=average_df, x="IR", y="AUC",
            color="black", linewidth=3, alpha=0.8, ax=ax, label="Mean"
        )

        # Ensure x-values match for fill_between
        x_values = np.arange(len(average_df["IR"]))  # Numeric values for x-axis
        ax.fill_between(
            x_values,
            average_df["AUC"] - std_dev,
            average_df["AUC"] + std_dev,
            color="gray", alpha=0.3
        )

        # Fix x-axis labels
        ax.set_xticks(x_values)
        ax.set_xticklabels(average_df["IR"])

    # Customize layout
    g.set_titles("")
    g.set_axis_labels("Imbalance Ratio (IR)", "AUC Score")
    g.tight_layout()

    # Show plot
    plt.show()
    """

    """
    # this is valid
    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Extract the mean and standard deviation
    average_df = df[["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(data["STD"])  # Convert to NumPy array for standard deviation

    # Set theme
    sns.set_theme(style="darkgrid")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot each dataset separately with a different color
    sns.lineplot(
        data=df_melted[df_melted["Dataset"].str.startswith("dataset")],
        x="IR", y="AUC", hue="Dataset", palette="tab10", linewidth=1, alpha=0.7
    )

    # Add the mean AUC trend in black
    sns.lineplot(
        data=average_df, x="IR", y="AUC",
        color="black", linewidth=3, linestyle="dashed", label="Average AUC score"
    )

    # Convert categorical x-axis labels to numerical indices for fill_between
    x_values = np.arange(len(average_df["IR"]))

    # Add standard deviation shading around the mean line
    plt.fill_between(
        x_values,
        average_df["AUC"] - std_dev,
        average_df["AUC"] + std_dev,
        color="gray", alpha=0.3, label="std"
    )

    # Adjust x-ticks
    plt.xticks(x_values, average_df["IR"])  # Set labels correctly

    # Labels and title
    plt.xlabel("Imbalance Ratio")
    plt.ylabel("AUC score")
    # plt.title("AUC Score Across Datasets and Imbalance Ratios")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()
    """

    """
    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Extract the mean and standard deviation
    average_df = df[["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(data["STD"])  # Convert to NumPy array for standard deviation

    # Set theme
    sns.set_theme(style="darkgrid")

    # Choose a color palette (e.g., "Set1" which will have consistent colors for datasets)
    palette = sns.color_palette("Set1", n_colors=len(df_melted["Dataset"].unique()) - 1)

    # Create the figure
    plt.figure(figsize=(8, 5))

    # Plot each dataset separately with a fixed color palette
    sns.lineplot(
        data=df_melted[df_melted["Dataset"].str.startswith("dataset")],
        x="IR", y="AUC", hue="Dataset", palette=palette, linewidth=1, alpha=0.7
    )

    # Plot the mean AUC trend using the same colors (we pick the first color from the palette)
    sns.lineplot(
        data=average_df, x="IR", y="AUC",
        color=palette[0], linewidth=3, label="Mean AUC"
    )

    # Convert categorical x-axis labels to numerical indices for fill_between
    x_values = np.arange(len(average_df["IR"]))

    # Add standard deviation shading around the mean line
    plt.fill_between(
        x_values,
        average_df["AUC"] - std_dev,
        average_df["AUC"] + std_dev,
        color="gray", alpha=0.3, label="Std Dev"
    )

    # Adjust x-ticks
    plt.xticks(x_values, average_df["IR"])  # Set labels correctly

    # Labels and title
    plt.xlabel("Imbalance Ratio (IR)")
    plt.ylabel("AUC Score")
    plt.title("AUC Score Across Datasets and Imbalance Ratios")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()
    """

    """
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Extract the mean and standard deviation
    average_df = df[["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(data["STD"])  # Convert to NumPy array for standard deviation

    # Set theme
    sns.set_theme(style="darkgrid")

    # Choose a color palette (e.g., "Set1" which will have consistent colors for datasets)
    palette = sns.color_palette("Set1", n_colors=len(df_melted["Dataset"].unique()) - 1)

    # Create the figure
    plt.figure(figsize=(8, 5))

    # Plot each dataset individually with a fixed color palette
    for idx, dataset in enumerate(df.columns[1:-2]):  # Skip the IR and Average columns
        sns.lineplot(
            data=df, x="IR", y=dataset,
            color=palette[idx], linewidth=1, alpha=0.7
        )

    # Plot the mean AUC trend using the same color (we pick the first color from the palette)
    sns.lineplot(
        data=average_df, x="IR", y="AUC",
        color=palette[0], linewidth=3, label="Mean AUC"
    )

    # Convert categorical x-axis labels to numerical indices for fill_between
    x_values = np.arange(len(average_df["IR"]))

    # Add standard deviation shading around the mean line
    plt.fill_between(
        x_values,
        average_df["AUC"] - std_dev,
        average_df["AUC"] + std_dev,
        color="gray", alpha=0.3, label="Std Dev"
    )

    # Adjust x-ticks
    plt.xticks(x_values, average_df["IR"])  # Set labels correctly

    # Labels and title
    plt.xlabel("Imbalance Ratio (IR)")
    plt.ylabel("AUC Score")
    plt.title("AUC Score Across Datasets and Imbalance Ratios")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()
    """

    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melted = df.melt(id_vars=["IR"], var_name="Dataset", value_name="AUC")

    # Extract the mean and standard deviation
    average_df = df[["IR", "Average"]].rename(columns={"Average": "AUC"})
    std_dev = np.array(data["STD"])  # Convert to NumPy array for standard deviation

    # Set theme
    sns.set_theme(style="darkgrid")

    plt.rcParams.update({'font.size': 40})

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
        # average_df["AUC"] - std_dev if average_df["AUC"] - std_dev > 0 else 0,
        low_boundary,
        average_df["AUC"] + std_dev,
        color="gray", alpha=0.2, label="Standard deviation"
    )

    # Adjust x-ticks
    plt.xticks(x_values, average_df["IR"])  # Set labels correctly

    # Labels and title
    plt.xlabel("Imbalance ratio")
    plt.ylabel("AUC score")
    # plt.title("AUC Score Across Datasets and Imbalance Ratios")

    # Show legend
    plt.legend()

    # Show the plot
    plt.savefig("./ir_graph_01.png", dpi=300, bbox_inches='tight')
    plt.show()
