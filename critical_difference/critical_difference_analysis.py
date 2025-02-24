import pandas as pd
import numpy as np
from critdd.diagram import Diagram
from scipy.stats import friedmanchisquare, norm

CLASSIFIERS = [
    "SVC_WEIGHTED",
    "SVC_SMOTE",
    "AdasynSVC",
    "XGBOOST_SMOTE",
    "Adacost",
    "SelfPaced",
    "TabNetMote",
    "UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT"
]


all = [
    'AdaBoost',
    'BBagg',
    'BRF',
    'EasyE',
    'IF',
    'LSAD',
    'OCSVM',
    'RF',
    'RUSBoost',
    'CBLOF',
    'COPOD',
    'KNN',
    'LMDD',
    'LOF',
    'MCD',
    'XGBOD',
    'LODA',
    'FBagg',
    'PAC',
    'SMOTE',
    'ADASYN',
    'BLSMOTE',
    'NearMiss',
    'TomekL',
    'ClusterC',
    'BCascade',
    'SelfPE',
    'UBagg',
    'OverBoost',
    'SMOTEBoost',
    'WSVC',
    'WRF',
    'SMOTEBagg',
    'AdaUBoost',
    'AdaCost'
]

sampling = [
    'ADASYN',
    'BLSMOTE',
    'SMOTE',
    'ClusterC',
    'NearMiss',
    'TomekL'
]

ensemble = [
    'AdaBoost',
    'RF'
]

outlier = [
    'CBLOF',
    'COPOD',
    'KNN',
    'LMDD',
    'LOF',
    'LSAD',
    'MCD',
    'OCSVM',
    'PCA'
]

hybrid_ensemble_sampling = [
    'BBagg',
    'BCascade',
    'BRF',
    'EasyE',
    'RUSBoost',
    'SelfPE',
    'UBagg',
    'OverBoost',
    'SMOTEBoost',
    'SMOTEBagg'
]

hybrid_ensemble_cost_sensitive =[
    'AdaCost',
    'AdaUBoost',
    'WRF'
]

hybrid_ensemble_outlier =[
    'FBagg',
    'IF',
    'LODA',
    'XGBOD'
]

clf_name_mapping={
    "sampling_adasyn_svc": ["ADASYN + SVC", "Sampling"],
    "sampling_adasyn_xgboost": ["ADASYN + XGBoost", "Sampling"],
    "sampling_borderlinesmote_svc": ["BLSMOTE + SVC", "Sampling"],
    "sampling_borderlinesmote_xgboost": ["BLSMOTE + XGBoost", "Sampling"],
    "sampling_smote_svc": ["SMOTE + SVC", "Sampling"],
    "sampling_smote_xgboost": ["SMOTE + XGBoost", "Sampling"],
    "sampling_cluster_centroids": ["ClusterC + SVC", "Sampling"],
    "sampling_cluster_centroids_xgboost": ["ClusterC + XGBoost", "Sampling"],
    "sampling_near_miss": ["NearMiss + SVC", "Sampling"],
    "sampling_near_miss_xgboost": ["NearMiss + XGBoost", "Sampling"],
    "sampling_tomek_links": ["TomekL + SVC", "Sampling"],
    "sampling_tomek_links_xgboost": ["TomekL + XGBoost", "Sampling"],
    "adaboost": ["AdaBoost", "Ensemble"],
    "random_forest": ["RF", "Ensemble"],
    "pyod_cblof": ["CBLOF", "Outlier"],
    "pyod_copod": ["COPOD", "Outlier"],
    "pyod_knn": ["KNN", "Outlier"],
    "pyod_lmdd": ["LMDD", "Outlier"],
    "pyod_lof": ["LOF", "Outlier"],
    "least_squares_anomaly_detection": ["LSAD", "Outlier"],
    "pyod_mcd": ["MCD", "Outlier"],
    "ocsvm": ["OCSVM", "Outlier"],
    "pyod_pca": ["PCA", "Outlier"],
    "weighted_svc": ["WSVC", "Cost-sensitive"],

    "balanced_bagging": ["BBagg", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_balanced_cascade": ["BCascade", "Hybrid (Ensemble, Sampling)"],
    "balanced_random_forest": ["BRF", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_easy_ensemble": ["EasyE", "Hybrid (Ensemble, Sampling)"],
    "rusboost": ["RUSBoost", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_self_paced": ["SelfPE", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_underbagging": ["UBagg", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_overboost": ["OverBoost", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_smote_boost": ["SMOTEBoost", "Hybrid (Ensemble, Sampling)"],
    "imbalanced_ensemble_smote_bagging": ["SMOTEBagg", "Hybrid (Ensemble, Sampling)"],

    "imbalanced_ensemble_adacost": ["AdaCost", "Hybrid (Ensemble, Cost-sensitive)"],
    "imbalanced_ensemble_adauboost": ["AdaUBoost", "Hybrid (Ensemble, Cost-sensitive)"],
    "weighted_random_forest": ["WRF", "Hybrid (Ensemble, Cost-sensitive)"],

    "pyod_feature_bagging": ["FBagg", "Hybrid (Ensemble, Outlier)"],
    "pyod_isolation_forest": ["IF", "Hybrid (Ensemble, Outlier)"],
    "pyod_loda": ["LODA", "Hybrid (Ensemble, Outlier)"],
    "pyod_xgbod": ["XGBOD", "Hybrid (Ensemble, Outlier)"],
    "sampling_combine_smote_enn_svc":["SMOTEENN + SVC","Sampling"],
    "sampling_combine_smote_enn_xgboost":["SMOTEENN + XGBoost","Sampling"],
    "sampling_combine_smote_tomek_svc":["SMOTETOMEK + SVC","Sampling"],
    "sampling_combine_smote_tomek_xgboost":["SMOTETOMEK + XGBoost","Sampling"]
}


def create_critical_difference_diagram(data, classifiers, output_filename):
    data['Clf_lower'] = data['Clf'].str.lower()
    classifiers_lower = [clf.lower() for clf in classifiers]

    # Now use isin with the lowercase versions
    mask = data['Clf_lower'].isin(classifiers_lower)
    # If you no longer need the 'Clf_lower' column, you can drop it
    data.drop('Clf_lower', axis=1, inplace=True)


    #data = data[mask]
    data = data.pivot(
        index="Dataset",
        columns="Classifier",
        values="geometric_mean"
    )
    data = data.dropna()
    # create a CD diagram from the Pandas DataFrame
    diagram = Diagram(
        data.to_numpy(),
        treatment_names=data.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    diagram.average_ranks  # the average rank of each treatment
    diagram.get_groups(alpha=.05, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        output_filename,
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": "critdd"},
    )
    return


_URL = "D:\_Research\Python\PyOD\\aggregated_g_mean_results.csv"
df = pd.read_csv(_URL)
df = df[df['Clf'].isin(clf_name_mapping)]
name_only_dict = {key: value[0] for key, value in clf_name_mapping.items()}
class_only_dict = {value[0]: value[1] for key, value in clf_name_mapping.items()}
df['Classifier'] = df['Clf'].replace(name_only_dict)

pivot_table = df.pivot(index='Dataset', columns='Classifier', values='geometric_mean').dropna()



stat, p = friedmanchisquare(*pivot_table.values.T)
print(f'Friedman test statistic: {stat}, p-value: {p}')

if p < 0.05:
    # Calculate ranks
    ranks = pivot_table.rank(axis=1, ascending=False)

    # Calculate average ranks of each classifier
    average_ranks = ranks.mean()

    # Number of datasets
    num_datasets = len(pivot_table)

    # Number of classifiers
    k = len(average_ranks)

    # Significance level
    alpha = 0.05

    # Critical Difference (CD) calculation
    CD = norm.ppf(1 - alpha / (2 * k * (k - 1))) * np.sqrt((k * (k + 1)) / (6 * num_datasets))

    print(f'Average Ranks: \n{average_ranks}')
    print(f'Critical Difference (CD): {CD}')

    axes = average_ranks.axes[0].values
    cds =  average_ranks.values

    df_final = pd.DataFrame(data={"Classifier":axes, "CD":cds})
    df_final['Classifier_Class'] = df_final['Classifier'].replace(class_only_dict)
    print(df_final[['Classifier_Class', 'Classifier', 'CD']])
    df_final.to_csv('cdiff.csv')

    # Interpretation:
    # Compare the difference in average ranks between any two classifiers.
    # If the difference exceeds the CD, the performance difference is considered statistically significant.
else:
    print("No significant differences were found among the classifiers.")

'''
create_critdd_diagram(df.copy(), sampling, 'critical_difference/sampling.pdf')
#ensemble sa neda spravit, lebo potrebuje aspon 3 klasifikatory
#create_critdd_diagram(df.copy(), ensemble, 'ensemble.pdf')
create_critdd_diagram(df.copy(), outlier, 'critical_difference/outlier.pdf')
create_critdd_diagram(df.copy(), hybrid_ensemble_sampling, 'critical_difference/hybrid_ensemble_sampling.pdf')
create_critdd_diagram(df.copy(), hybrid_ensemble_cost_sensitive, 'critical_difference/hybrid_ensemble_cost_sensitive.pdf')
create_critdd_diagram(df.copy(), hybrid_ensemble_outlier, 'critical_difference/hybrid_ensemble_outlier.pdf')
create_critdd_diagram(df.copy(), all, 'critical_difference/all.pdf')
'''

create_critdd_diagram(df.copy(), all, 'critical_difference/all.pdf')