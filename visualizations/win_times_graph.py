import matplotlib.pyplot as plt
import seaborn
import pandas as pd


if __name__ == "__main__":
    # Sample data
    data = {
        'Method': ['W-SVM', 'SMOTE+SVM', 'ADASYN+SVM', 'SMOTE+XGBoost', 'AdaCost', 'SelfPE', 'SMOTE+TabNet',
                   "GA-HESO"],
        'AUC': [3, 2, 0, 2, 4, 4, 1, 13],
        'GM': [3, 2, 0, 2, 4, 4, 1, 13]
    }

    df = pd.DataFrame(data)

    # Reshape data for better visualization
    df_melted = df.melt(id_vars='Method', var_name='Metric', value_name='Wins')

    # Plot
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(14, 8))
    seaborn.barplot(x='Method', y='Wins', hue='Metric', data=df_melted, palette=['red', 'blue'])
    # plt.xlabel("Used Methods")
    plt.ylabel("Number of Wins")
    # plt.title("Winning Times for Utilized Methods")
    plt.legend(title="Score Type")
    plt.xticks(rotation=40)

    plt.savefig("./win_rate_final.png", dpi=300, bbox_inches='tight')
    plt.show()
