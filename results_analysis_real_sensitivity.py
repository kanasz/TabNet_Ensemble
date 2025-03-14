import numpy as np
import pandas as pd
import pygad
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
import os

from constants import WEAK_CLASSIFIERS_COUNT

def split_array(selected_samples, samples_counts):
    subarrays = []
    start = 0
    for count in samples_counts:
        subarrays.append(selected_samples[start:start + count])
        start += count
    return subarrays

base_folder = 'predictions/{}/results_sensitivity'
model_name = 'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_{}_{}'
datasets = [('abalone','abalone_20_vs_8_9_10'),
            ('ecoli','ecoli_0_vs_1'),
            ('glass','glass_5'),
            ('wine','red_8_vs_6'),
            ('yeast','yeast_3')]
#datasets = [('ecoli','ecoli_0_vs_1')]
#datasets = [('glass','glass_5')]
#datasets = [('wine','red_8_vs_6')]
#datasets = [('yeast','yeast_3')]
classifiers_count = ['2','4','6','8','10','14','18']

dataset_folds_counts = {
    'abalone':[17, 15, 18, 18, 16],
    'ecoli':[27, 25, 26, 28, 29],
    'glass':[26, 25, 27, 27, 31],
    'wine':[25, 22, 25, 31, 29],
    'yeast':[18, 17, 15, 17, 24]
}

def process_ga(path):
    ga_instance = pygad.load(path)
    best_solution = ga_instance.best_solutions[-1]
    best_fitness = ga_instance.best_solutions_fitness[-1]
    return best_solution, best_fitness
def process_results(path):
    # Open the file in read mode
    with open(path, 'r') as file:
        # Read the content of the file
        file_content = file.read()

    file_content = file_content.replace(" ","").replace("\n","")
    file_content = (file_content.replace("array(","")
                    .replace(")","")
                    .replace(",dtype=int64", "")
                    .replace("dtype=int64,","")
                    .replace(",dtype=object","")
                    )
    # Evaluate the content to convert it to a dictionary
    data = eval(file_content)

    gmeans = []
    roc_aucs = []
    if data['true_values']==None:
        return "0"
    for idx, true_values in enumerate(data['true_values']):
        predicions = data['predicted_values'][idx]
        gmean = geometric_mean_score(true_values, predicions)
        roc_auc = roc_auc_score(true_values, predicions)
        roc_aucs.append(roc_auc)
        gmeans.append(gmean)
    return "{}".format(np.average(gmeans))
all_counts = None
all_counts_unaggregated = None
for ds in datasets:
    ds_base_folder = base_folder.format(ds[0])
    array = None
    selected_samples = None
    for clf in classifiers_count:
        file_name = model_name.format(clf, ds[1])
        file_path = "{}/{}".format(ds_base_folder, file_name)
        #print("ds[1]")
        results = process_results(file_path+".txt")
        best_solution, best_fitness = process_ga(file_path)
        clf_count = np.sum(best_solution[0:WEAK_CLASSIFIERS_COUNT])
        samples_count = np.sum(best_solution[WEAK_CLASSIFIERS_COUNT:])
        if array is None:
            array =best_solution[0:WEAK_CLASSIFIERS_COUNT]
        else:
            array = np.vstack((array, best_solution[0:WEAK_CLASSIFIERS_COUNT]))
        if selected_samples is None:
            selected_samples = best_solution[WEAK_CLASSIFIERS_COUNT:]
        else:
            selected_samples = np.vstack((selected_samples, best_solution[WEAK_CLASSIFIERS_COUNT:]))
        print("{} {}: {} / {}".format(ds[1], clf, results, clf_count))


        samples_counts = dataset_folds_counts[ds[0]]
        selected_samples = best_solution[WEAK_CLASSIFIERS_COUNT:]
        splitted = split_array(selected_samples, samples_counts)
        print("{}/{}\t{}/{}\t{}/{}\t{}/{}\t{}/{}".format(
            np.sum(splitted[0]),len(splitted[0]),
            np.sum(splitted[1]), len(splitted[1]),
            np.sum(splitted[2]), len(splitted[2]),
            np.sum(splitted[3]), len(splitted[3]),
            np.sum(splitted[4]), len(splitted[4])
        ))

        #print("{} {}: {}/{}".format(ds[1], clf, samples_count, len(best_solution[WEAK_CLASSIFIERS_COUNT:])))

    #print(array)
    clf_sum = (np.sum(array, axis=0))
    if all_counts_unaggregated is None:
        all_counts_unaggregated = array
    else:
        all_counts_unaggregated = np.vstack((all_counts_unaggregated, array))
    print(clf_sum)
    if all_counts is None:
        all_counts = clf_sum
    else:
        all_counts = np.vstack((all_counts, clf_sum))


    print("---------------------------------------------")
print(all_counts)

column_names = ['Column1', 'Column2', 'Column3', 'Column4']

# Create a pandas DataFrame
df = pd.DataFrame(all_counts)
df_unaggregated = pd.DataFrame(all_counts_unaggregated)
df.to_csv("counts.csv")
df_unaggregated.to_csv("unaggregated.csv")