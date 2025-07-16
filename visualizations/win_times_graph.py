import matplotlib.pyplot as plt
import seaborn
import pandas as pd

if __name__ == "__main__":
    # win rate synthetic data + real data
    data = {
        'Method': ['W-SVM', 'SMOTE+SVM', 'ADASYN+SVM', 'SMOTE+XGBoost', 'AdaCost', 'SelfPE', 'SMOTE+TabNet',
                   "GA-HESO"],
        'AUC': [3, 2, 0, 2, 5, 4, 2, 14],
        'GM': [3, 2, 0, 2, 5, 4, 2, 14]
    }

    df = pd.DataFrame(data)
    # Reshape data for better visualization
    df_melted = df.melt(id_vars='Method', var_name='Metric', value_name='Wins')

    # Plot
    plt.figure(figsize=(14, 8))
    seaborn.barplot(x='Method', y='Wins', hue='Metric', data=df_melted, palette=['red', 'blue'])
    plt.ylabel("Number of Wins", fontsize=20)
    plt.xlabel("")
    plt.xticks(fontsize=20)  # X-axis tick labels
    plt.yticks(fontsize=20)  # Y-axis tick labels
    plt.legend(fontsize=20)
    plt.xticks(rotation=40)

    plt.savefig("./win_rate_both_data_graph.png", dpi=300, bbox_inches='tight')
    plt.show()
