import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("SMOTE_MS_results_by_folds.csv")
df['total'] = df['selected_oversampled'] + df['not_selected_oversampled']
df = df.groupby(['name']).mean().sort_values('total', ascending=False).reset_index()
print(df)

f, ax = plt.subplots(figsize = (10,8))
sns.set_color_codes('pastel')
sns.barplot(x = 'total', y = 'name', data = df,
            label = 'Total samples', color = 'b', edgecolor = 'w')
sns.set_color_codes('muted')
sns.barplot(x = 'selected_oversampled', y = 'name', data = df,
            label = 'Selected samples', color = 'b', edgecolor = 'w')
legend = ax.legend(ncol = 2, loc = 'lower center', bbox_to_anchor=(0.5, -0.13),fontsize = 12)
# Set transparent background and remove the border
legend.get_frame().set_alpha(0)  # Make background transparent
legend.get_frame().set_linewidth(0)  # Remove border
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel('')
ax.set_xlabel('')
plt.subplots_adjust(left=0.27)
sns.despine(left = True, bottom = True)
plt.savefig("total_vs_selected.png")
plt.show()