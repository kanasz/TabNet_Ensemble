#numbers of clusters
import os
import random
import numpy as np
import torch
import pandas as pd
import pygad
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from base_functions import get_yeast_3_data, get_yeast_4_data, get_meanshift_cluster_counts, \
    get_meanshift_cluster_counts_reporting, get_yeast_6_data, get_yeast_5_data, get_abalone_3_vs_11_data, \
    get_abalone_9_vs_18_data, get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, \
    get_wine_quality_white_9_vs_4_data, get_wine_quality_white_3_vs_7_data, get_wine_quality_red_8_vs_6_data, \
    get_wine_quality_red_3_vs_5_data, get_dbscan_cluster_counts_for_reporting, get_ecoli_0_vs_1_data, \
    get_ecoli_0_4_6_vs_5_data, get_ecoli_0_3_4_vs_5_data, get_ecoli_0_2_3_4_vs_5_data, get_glass_2_data, \
    get_glass_4_data, get_glass_5_data, get_glass_0_1_6_vs_5_data, get_synthetic_data, get_sensitivity_synthetic_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from imblearn.over_sampling import ADASYN
from constants import SYNTHETIC_MINORITY_COUNT, SMOTE_K_NEIGHBORS

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.rand(seed)
random.SystemRandom(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multiGPUs.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#selected clusters

#number of samples in selected clusters


cmap1 = mcolors.ListedColormap(['red', 'blue', 'green'])
cmap2 = mcolors.ListedColormap(['red', 'blue', 'green', 'orange'])
experiment_name = "SMOTE_MS"
#experiment_name = "SMOTE_DBSCAN"
#experiment_name = "ADASYN_MS"

classifiers = [
                'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
                #'UNCLUSTERED_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
                #'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_DBSCAN'
               ]

def process_results(path, name, clusters, synthetic, X,y,  fold = 0, experiment_name = "SMOTE_MS"):
    row = {}
    ga_instance = pygad.load(path)
    solution = ga_instance.best_solutions[-1]
    samples_solution = solution[35:]
    '''
    if numerical_cols is None:
        numerical_cols = list(data[0].columns.values)

    
    if "SMOTE" in experiment_name:
        if name=='white_9_vs_4' and "SMOTE" in experiment_name:
            sampling_algorithm = SMOTE(random_state=42, k_neighbors=3)
        else:
            sampling_algorithm = SMOTE(random_state=42)
    if "ADASYN" in experiment_name:
        sampling_strategy = {1: sum(data[1] == 1) + SYNTHETIC_MINORITY_COUNT}
        n_neighbors = SMOTE_K_NEIGHBORS
        sampling_algorithm = ADASYN(sampling_strategy=sampling_strategy,
                                    random_state=42, n_neighbors=n_neighbors)

    if "MS" in experiment_name:
        clusters, bandwidths, algs, X, y, synthetic, clusters_labels = get_meanshift_cluster_counts_reporting(data[0], data[1], numerical_cols, categorical_cols,
                                                                  smote=sampling_algorithm)
    elif "DBSCAN" in experiment_name:
        clusters, algs, X, y, synthetic, clusters_labels = get_dbscan_cluster_counts_for_reporting(data[0],
                                                                                                              data[1],
                                                                                                              numerical_cols,
                                                                                                              categorical_cols,
                                                                                                              smote=sampling_algorithm)
    '''
    row['fold'] = fold
    row['name'] = name
    row['majority'] = 0
    row['minority'] = 0
    row['selected_oversampled'] = 0
    row['not_selected_oversampled'] = 0
    row['clusters_count'] = 0
    row['selected_clusters'] = 0

    start_indices = np.cumsum([0] + clusters[:-1]) + 35
    end_indices = np.cumsum(clusters) + + 35
    selected = np.array(solution[start_indices[fold]:end_indices[fold]])
    if len(selected) < np.max(clusters_labels[fold])+1:
        to_add = np.max(clusters_labels[fold]) - len(selected) +1
        selected = np.hstack([selected, np.zeros(to_add + 1)])
    try:
        selected_mask = selected[np.array(clusters_labels[fold])] == 1
    except:
        print("error")
    row['clusters_count'] = len(selected)
    row['selected_clusters'] = np.sum(selected)

    data_X = np.vstack([synthetic[fold], X[fold]])
    data_y = np.hstack([np.ones(synthetic[fold].shape[0]) * 2, np.array(y[fold])])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(data_X)

    # Create a figure with 2 subplots side by side
    '''
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    '''
    data_y_1 = data_y.copy()
    # Plot 1: All Data

    '''
    axes[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=data_y_1, cmap=cmap1, alpha=0.5)
    axes[0].set_title('t-SNE Visualization (All Data) - {}'.format(name))
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    '''
    # Prepare the filtered data mask
    filtered_data = np.concatenate((selected_mask, np.array([True] * len(X[0]))))
    if len(data_y) > len(filtered_data):
        data_y = data_y[:len(filtered_data)-1]
        tsne_results = tsne_results[:len(filtered_data)-1]

    if len(data_y) < len(filtered_data):
        filtered_data = filtered_data[:len(filtered_data)-1]
        #data_y = data_y[:len(filtered_data)-1]
        filtered_data = filtered_data[:len(data_y)]

    data_y[~filtered_data] = 3
    # Plot 2: Filtered Data
    #axes[1].scatter(tsne_results[filtered_data][:, 0], tsne_results[filtered_data][:, 1], c=data_y[filtered_data], cmap=cmap, alpha=0.5)
    '''
    axes[1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=data_y,
                    cmap=cmap2, alpha=0.5)
    axes[1].set_title('t-SNE Visualization (GA Optimized) - {}'.format(name))
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    '''
    row['majority'] = np.sum(data_y ==0)
    row['minority'] = np.sum(data_y ==1)
    row['selected_oversampled'] = np.sum(data_y ==2)
    row['not_selected_oversampled'] = np.sum(data_y ==3)

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Majority', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Minority', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='All Oversampled', markerfacecolor='green', markersize=10)
    ]

    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', label='Majority', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Minority', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Selected Oversampled', markerfacecolor='green',
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Not selected Oversampled', markerfacecolor='orange',
               markersize=10)
    ]

    # Add legend to the first subplot
    #axes[0].legend(handles=legend_elements, title='Classes', loc='best')

    # Add legend to the second subplot (optional)
    #axes[1].legend(handles=legend_elements2, title='Classes', loc='best')

    # Adjust layout for better spacing
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data_y,
                    cmap=cmap2, alpha=0.5)
    plt.title('t-SNE Visualization (GA Optimized) - {}'.format(name))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=legend_elements2, title='Classes', loc='best')
    plt.tight_layout()

    # Save the combined figure
    plt.savefig("images/{}_{}_comparison_fold_{}.png".format(experiment_name, name,fold))
    plt.clf()

    # Show the plot
    #plt.show()

    print("NAME: {}, {}".format(name, clusters))

    return row








folder_path = "D:\\_Research\\Python\\Boosting_Tabnet\\predictions\\{}\\results"
results_folders = {
    #"synthetic":[ 'synthetic_04'],
    #"synthetic":['synthetic_01', 'synthetic_02', 'synthetic_04'],
    #"abalone":['abalone_3_vs_11','abalone_9_vs_18','abalone_19_vs_10_11_12_13','abalone_20_vs_8_9_10'],
    #"wine":['white_9_vs_4', 'white_3_vs_7','red_8_vs_6','red_3_vs_5'],
    #"ecoli":['ecoli_0_vs_1','ecoli_0_4_6_vs_5','ecoli_0_3_4_vs_5','ecoli_0_2_3_4_vs_5'],
    #"glass": ['glass_2', 'glass_4', 'glass_5', 'glass_0_1_6_vs_5'],
    #'yeast':['yeast_3','yeast_4','yeast_5','yeast_6'],
    'synthetic':['synthetic_01', 'synthetic_02', 'synthetic_04']
}




data = {
    'yeast_3' : {
        "data": get_yeast_3_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'yeast_4' : {
        "data": get_yeast_4_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'yeast_5' : {
        "data": get_yeast_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'yeast_6' : {
        "data": get_yeast_6_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'abalone_3_vs_11' : {
        "data": get_abalone_3_vs_11_data(),
        "numerical_cols": ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        "categorical_cols":['Sex']
    },
    'abalone_9_vs_18' : {
        "data": get_abalone_9_vs_18_data(),
        "numerical_cols": ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        "categorical_cols":['Sex']
    },
    'abalone_19_vs_10_11_12_13' : {
        "data": get_abalone_19_vs_10_11_12_13_data(),
        "numerical_cols": ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        "categorical_cols":['Sex']
    },
    'abalone_20_vs_8_9_10' : {
        "data": get_abalone_20_vs_8_9_10_data(),
        "numerical_cols": ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        "categorical_cols":['Sex']
    },
    'white_9_vs_4' : {
        "data": get_wine_quality_white_9_vs_4_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'white_3_vs_7' : {
        "data": get_wine_quality_white_3_vs_7_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'red_8_vs_6' : {
        "data": get_wine_quality_red_8_vs_6_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    'red_3_vs_5' : {
        "data": get_wine_quality_red_3_vs_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "ecoli_0_vs_1":{
        "data": get_ecoli_0_vs_1_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "ecoli_0_4_6_vs_5":{
        "data": get_ecoli_0_4_6_vs_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "ecoli_0_3_4_vs_5":{
        "data": get_ecoli_0_3_4_vs_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "ecoli_0_2_3_4_vs_5":{
        "data": get_ecoli_0_2_3_4_vs_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "glass_2":{
        "data": get_glass_2_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "glass_4":{
        "data": get_glass_4_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "glass_5":{
        "data": get_glass_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "glass_0_1_6_vs_5":{
        "data": get_glass_0_1_6_vs_5_data(),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "synthetic_1":{
        "data": get_synthetic_data("01", "0.3", 50),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "synthetic_2":{
        "data": get_synthetic_data("02", "0.1", 100,200),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "synthetic_4":{
        "data": get_sensitivity_synthetic_data("01", "0.05", 20, 250),
        "numerical_cols": None,
        "categorical_cols":None
    }
}

data = {
    "synthetic_01":{
        "data": get_synthetic_data("01", "0.3", 50),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "synthetic_02":{
        "data": get_synthetic_data("02", "0.1", 100,200),
        "numerical_cols": None,
        "categorical_cols":None
    },
    "synthetic_04":{
        "data": get_sensitivity_synthetic_data("01", "0.05", 20, 250),
        "numerical_cols": None,
        "categorical_cols":None
    }
}

results = {
    "fold":[],
    "name":[],
    "majority":[],
    "minority":[],
    "selected_oversampled":[],
    "not_selected_oversampled":[],
    "clusters_count":[],
    "selected_clusters":[]
}
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
                        print("{}:{}".format(classifier, dataset))
                        if os.path.exists(file_path):
                            if dataset in data.keys():
                                if dataset=="abalone_3_vs_11" and experiment_name=="ADASYN_MS":
                                    continue

                                value_data= data[dataset]['data']
                                if "SMOTE" in experiment_name:
                                    if dataset == 'white_9_vs_4' and "SMOTE" in experiment_name:
                                        sampling_algorithm = SMOTE(random_state=42, k_neighbors=3)
                                    else:
                                        sampling_algorithm = SMOTE(random_state=42)
                                if "ADASYN" in experiment_name:
                                    sampling_strategy = {1: sum(value_data[1] == 1) + SYNTHETIC_MINORITY_COUNT}
                                    n_neighbors = SMOTE_K_NEIGHBORS
                                    sampling_algorithm = ADASYN(sampling_strategy=sampling_strategy,
                                                                random_state=42, n_neighbors=n_neighbors)
                                numerical_cols = data[dataset]['numerical_cols']
                                categorical_cols = data[dataset]['categorical_cols']
                                if numerical_cols is None:
                                    numerical_cols = list(value_data[0].columns.values)
                                if "MS" in experiment_name:
                                    clusters, bandwidths, algs, X, y, synthetic, clusters_labels = get_meanshift_cluster_counts_reporting(
                                        value_data[0], value_data[1], numerical_cols, categorical_cols,
                                        smote=sampling_algorithm)
                                elif "DBSCAN" in experiment_name:
                                    clusters, algs, X, y, synthetic, clusters_labels = get_dbscan_cluster_counts_for_reporting(
                                        value_data[0],
                                        value_data[1],
                                        numerical_cols,
                                        categorical_cols,
                                        smote=sampling_algorithm)




                                for fold in range(5):

                                    result = [process_results(file_path.replace('.pkl',''), dataset,clusters, synthetic, X,y
                                                                       , fold, experiment_name)]

                                    results["fold"].append(result[0]["fold"])
                                    results["name"].append(result[0]["name"])
                                    results["majority"].append(result[0]["majority"])
                                    results["minority"].append(result[0]["minority"])
                                    results["selected_oversampled"].append(result[0]["selected_oversampled"])
                                    results["not_selected_oversampled"].append(result[0]["not_selected_oversampled"])
                                    results["clusters_count"].append(result[0]["clusters_count"])
                                    results["selected_clusters"].append(result[0]["selected_clusters"])
                        print("------------------------------------")
        new_row = pd.DataFrame(row)
        #df = pd.concat([df, new_row], ignore_index=True)

df = pd.DataFrame(results)


df.to_csv("{}_synthetic_results_by_folds.csv".format(experiment_name),index=None)

print(df)