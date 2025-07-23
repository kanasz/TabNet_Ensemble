import os
import seaborn as sns
import numpy as np
import pandas as pd
import pygad
from matplotlib import pyplot as plt

from base_functions import get_yeast_3_data

folder_path = "D:\\_Research\\Python\\Boosting_Tabnet\\predictions\\{}\\results"
results_folders = {
    "abalone":['abalone_3_vs_11','abalone_9_vs_18','abalone_19_vs_10_11_12_13','abalone_20_vs_8_9_10'],
    "wine":['white_9_vs_4', 'white_3_vs_7','red_8_vs_6','red_3_vs_5'],
    "ecoli":['ecoli_0_vs_1','ecoli_0_4_6_vs_5','ecoli_0_3_4_vs_5','ecoli_0_2_3_4_vs_5'],
    "glass": ['glass_2', 'glass_4', 'glass_5', 'glass_0_1_6_vs_5'],
    'yeast':['yeast_3','yeast_4','yeast_5','yeast_6'],
    'synthetic':['synthetic_01', 'synthetic_02', 'synthetic_04']
}

data = [
    'yeast_3', 'yeast_4','yeast_5', 'yeast_6',
    'abalone_3_vs_11', 'abalone_9_vs_18', 'abalone_19_vs_10_11_12_13', 'abalone_20_vs_8_9_10',
    'white_9_vs_4', 'white_3_vs_7', 'red_8_vs_6', 'red_3_vs_5',
    "ecoli_0_vs_1",
    "ecoli_0_4_6_vs_5",
    "ecoli_0_3_4_vs_5",
    "ecoli_0_2_3_4_vs_5",
    "glass_2",
    "glass_4",
    "glass_5",
    "glass_0_1_6_vs_5",
    "synthetic_01",
    "synthetic_02",
    "synthetic_04"
]

'''
data = [
    "synthetic_01",
    "synthetic_02",
    "synthetic_04"
]
'''
classifiers = [
                'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
                #'UNCLUSTERED_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
                #'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_DBSCAN'
               ]

experiment_name = "SMOTE_MS"
#experiment_name = "SMOTE_DBSCAN"
#experiment_name = "ADASYN_MS"

dict_results = {
    "dataset":[],
    "count":[]
}
'''
for folder in results_folders.keys():
    for dataset in results_folders[folder]:
        row = {
            "dataset":[dataset]
        }
        results_path  = folder_path.format(folder)
        for classifier in classifiers:

            file = str.upper("{}_{}".format(classifier, dataset))
            for file_name in os.listdir(results_path):
                if file_name.endswith('.pkl'):
                    if file in str.upper(file_name):
                        file_path = os.path.join(results_path, file_name)
                        #print("{}:{}".format(classifier, dataset))
                        if os.path.exists(file_path):
                            if dataset in data:
                                if dataset=="abalone_3_vs_11" and experiment_name=="ADASYN_MS":
                                    continue
                                ga_instance = pygad.load(file_path.replace('.pkl',''))
                                solution = ga_instance.best_solutions[-1]
                                samples_solution = solution[:35]
                                clf_sum = np.sum(samples_solution)
                                print("{}, {}:{}".format(classifier,dataset, clf_sum))
                                dict_results["dataset"].append(dataset)
                                dict_results["count"].append(clf_sum)

df = pd.DataFrame(dict_results)
df.to_csv("selected_classifiers_count.csv")
'''
df = pd.read_csv("SMOTE_MS_results_by_folds.csv")
df['total'] = df['selected_oversampled'] + df['not_selected_oversampled']
df1 = df.groupby(['name'],sort=False).mean().reset_index()

df2 = pd.read_csv("selected_classifiers_count.csv")

print(df2["dataset"])
print(df1["name"])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 12))

sns.set_color_codes('pastel')

# First plot: normal bar plot
sns.barplot(x = 'name', y = 'total', data = df,
            label = 'Total samples', color = 'b', edgecolor = 'w', ax=ax1, errorbar=None)
sns.set_color_codes('muted')
sns.barplot(x = 'name', y = 'selected_oversampled', data = df,
            label = 'Selected samples', color = 'b', edgecolor = 'w', ax=ax1, errorbar=None)

ax1.legend(loc='upper right', fontsize=18)
ax1.set_ylabel('Count', fontsize=18)
#ax1.set_title('First Data')
# Force x-axis tick labels to show on ax1 if desired:
ax1.tick_params(axis='x', labelrotation=90, labelsize=18)

# Second plot: mirrored bar plot
sns.barplot(x='dataset', y='count', data=df2,
            label='Selected classifiers count', color='r', edgecolor='w', ax=ax2)
ax2.invert_yaxis()  # Mirror the second plot
ax2.legend(loc='upper right', fontsize=18)
ax2.set_ylabel('Count', fontsize=18)
#ax2.set_title('Second Data (Mirrored)')
ax2.tick_params(axis='x', labelrotation=90, labelsize=18)

# Set an overall x-axis label on the bottom subplot (visible for the entire figure)
ax2.set_xlabel('')
ax2.set_yticks([0,5,10,15,20])
plt.tight_layout()
plt.savefig("selected_samples_and_classifiers.png", bbox_inches='tight')
plt.show()