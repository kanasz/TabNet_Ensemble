import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from constants import LossFunction, WEAK_CLASSIFIERS_COUNT, SMOTE_K_NEIGHBORS, Classifier, RANDOM_STATE
from loss_functions.binary_vs_loss import BinaryVSLoss
from loss_functions.binary_vs_loss_mdr import BinaryVSLossMDR
from loss_functions.cross_entropy_loss import CrossEntropyLoss
from loss_functions.ib_focal_loss import IBFocalLoss
from loss_functions.ib_loss import IBLoss
from loss_functions.ib_loss_mdr import IBLossMDR
from loss_functions.ldam_loss import LDAMLoss
from loss_functions.ldam_loss_mdr import LDAMLossMDR
from loss_functions.vs_loss import VSLoss
from loss_functions.vs_loss_mdr import VSLossMDR
import imbalanced_ensemble.ensemble as imb

# from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

def get_slovak_data(business_area, year, postfix):
    # print("Loading Slovak data...")
    path_bankrupt = Path(__file__).parent / "data/slovak_data/parsed_data/bankrupt/bankrupt_{}_{}_year_{}.csv" \
        .format(business_area, year, postfix)
    path_non_bankrupt = Path(__file__).parent / "data/slovak_data/parsed_data/non_bankrupt/nonbankrupt_{}_{}_year_{}" \
                                                ".csv".format(business_area, year, postfix)
    # print("Data: {}".format(path_bankrupt))
    bankrupt_data = pd.read_csv(path_bankrupt)
    non_bankrupt_data = pd.read_csv(path_non_bankrupt)
    # features = bankrupt_data.drop(["IS_BANKRUPT"], axis=1).append(non_bankrupt_data.drop(["IS_BANKRUPT"], axis=1))
    # labels = bankrupt_data["IS_BANKRUPT"].append(non_bankrupt_data["IS_BANKRUPT"])

    features = pd.concat([bankrupt_data.drop(["IS_BANKRUPT"], axis=1), non_bankrupt_data.drop(["IS_BANKRUPT"], axis=1)])
    labels = pd.concat([bankrupt_data["IS_BANKRUPT"], non_bankrupt_data["IS_BANKRUPT"]])
    # print("Info: rows - {}, columns - {}".format(len(features), len(features.columns)))
    return features, labels


def get_synthetic_data(prefix, contamination, features, samples=None):
    if samples is None:
        path = Path(__file__).parent / "data/synthetic_data/{}_synthetic_{}_contamination_{}_features.csv" \
            .format(prefix, contamination, features)
    else:
        path = Path(__file__).parent / "data/synthetic_data/{}_synthetic_{}_contamination_{}_features_{}_samples.csv" \
            .format(prefix, contamination, features, samples)

    data = pd.read_csv(path)
    features = data.drop(["target"], axis=1)
    labels = data['target']
    return features, labels


def get_scoring_dict():
    scoring_dict = {
        # 'accuracy_score': make_scorer(accuracy_score),
        # 'f1_score': make_scorer(f1_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'geometric_mean_score': make_scorer(geometric_mean_score),
        # 'sensitivity_score': make_scorer(sensitivity_score),
        # 'specificity_score': make_scorer(specificity_score)
    }
    return scoring_dict


def save_results(clf, data_origin, file_name, best_params, cv_results):
    print("Saving prediction results...")
    df = pd.DataFrame()
    for param in cv_results:
        if param.find('train') > -1:
            continue
        df[param] = cv_results[param]

    columns_to_drop = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params']
    df = df.drop(columns=columns_to_drop)
    path = Path(__file__).parent / "predictions/{}/results/{}/".format(clf, data_origin)
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, file_name + "_results.csv"))
    path = Path(__file__).parent / "predictions/{}/results/{}/{}_best_params.csv".format(clf, data_origin, file_name)
    w = csv.writer(open(path, "w"))
    for key, val in best_params.items():
        w.writerow([key.replace('model__', ''), val])
    return


def get_fraudulent_claim_on_cars_physical_damage_data():
    path = Path(__file__).parent / "data/fraudulent_claim_on_cars_physical_damage/training data subsampled.csv"
    data = pd.read_csv(path)
    data = data.dropna()
    features = data.drop(["fraud", "claim_number", "zip_code", "claim_date", "claim_day_of_week"], axis=1)
    labels = data['fraud']
    return features, labels

def get_glass_0_1_6_vs_5_data():
    path = Path(__file__).parent / "data/glass/glass-0-1-6_vs_5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']
def get_glass_2_data():
    path = Path(__file__).parent / "data/glass/glass2.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_glass_4_data():
    path = Path(__file__).parent / "data/glass/glass4.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_glass_5_data():
    path = Path(__file__).parent / "data/glass/glass5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_yeast_3_data():
    path = Path(__file__).parent / "data/yeast/yeast3.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_yeast_4_data():
    path = Path(__file__).parent / "data/yeast/yeast4.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_yeast_5_data():
    path = Path(__file__).parent / "data/yeast/yeast5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_yeast_6_data():
    path = Path(__file__).parent / "data/yeast/yeast6.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_ecoli_0_vs_1_data():
    path = Path(__file__).parent / "data/ecoli/ecoli-0_vs_1.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_ecoli_0_2_3_4_vs_5_data():
    path = Path(__file__).parent / "data/ecoli/ecoli-0-2-3-4_vs_5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_ecoli_0_3_4_vs_5_data():
    path = Path(__file__).parent / "data/ecoli/ecoli-0-3-4_vs_5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_ecoli_0_4_6_vs_5_data():
    path = Path(__file__).parent / "data/ecoli/ecoli-0-4-6_vs_5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']

def get_wine_quality_red_3_vs_5_data():
    path = Path(__file__).parent / "data/wine/winequality-red-3_vs_5.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_wine_quality_red_8_vs_6_data():
    path = Path(__file__).parent / "data/wine/winequality-red-8_vs_6.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_wine_quality_white_3_vs_7_data():
    path = Path(__file__).parent / "data/wine/winequality-white-3_vs_7.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_wine_quality_white_9_vs_4_data():
    path = Path(__file__).parent / "data/wine/winequality-white-9_vs_4.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_abalone_9_vs_18_data():
    path = Path(__file__).parent / "data/abalone/abalone9-18.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_abalone_3_vs_11_data():
    path = Path(__file__).parent / "data/abalone/abalone-3_vs_11.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_abalone_19_vs_10_11_12_13_data():
    path = Path(__file__).parent / "data/abalone/abalone-19_vs_10-11-12-13.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_abalone_20_vs_8_9_10_data():
    path = Path(__file__).parent / "data/abalone/abalone-20_vs_8-9-10.dat"
    df = load_keel_dat_file(path)
    features = df.drop(["Class"], axis=1)
    df['Class'] = df['Class'].str.strip().str.lower()
    df['Class'] = df['Class'].replace({'positive': 1, 'negative': 0})
    return features, df['Class']


def get_aids_data(features):
    path = Path(__file__).parent / "data/aids_classification/aids_classification_{}.csv".format(features)
    data = pd.read_csv(path)
    data = data.dropna()
    features = data.drop(["infected"], axis=1)
    labels = data['infected']
    return features, labels


def get_taiwan_bankruptcy_data():
    _path = Path(__file__).parent / "data/taiwan_data/taiwan_parsed_data.csv"
    _data = pd.read_csv(_path)
    _taiwan_features = _data.drop(["is_bankrupt"], axis=1)
    _taiwan_labels = _data['is_bankrupt']
    return _taiwan_features, _taiwan_labels


def get_polish_bankruptcy_data(index: int):
    _path = Path(__file__).parent / "data/polish_data/polish_{}year.csv".format(index)
    _data = pd.read_csv(_path)
    _polish_features = _data.drop(['is_bankrupt'], axis=1)
    _polish_labels = _data['is_bankrupt']
    return _polish_features, _polish_labels


def resample_minority_samples(X_train, y_train, selected_resampled=None, syntetic_minority_count=100,
                              cluster_count=30):
    smote = SMOTE(sampling_strategy={1: sum(y_train == 1) + syntetic_minority_count},
                  random_state=42, k_neighbors=SMOTE_K_NEIGHBORS)  # Assuming the minority class label is 1
    X_res, y_res = smote.fit_resample(X_train, y_train)

    n_samples_original = X_train.shape[0]
    X_synthetic = X_res[n_samples_original:]
    y_synthetic = y_res[n_samples_original:]

    X_synthetic = X_synthetic[:syntetic_minority_count]
    y_synthetic = y_synthetic[:syntetic_minority_count]

    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    kmeans.fit(X_synthetic)
    if type(selected_resampled) is list:
        selected_resampled = np.array(selected_resampled)

    X_reduced_synthetic = kmeans.cluster_centers_
    y_reduced_synthetic = np.full(shape=cluster_count, fill_value=1)  # Assuming the minority class is labeled as 1
    X_reduced_synthetic = X_reduced_synthetic[selected_resampled == True]
    y_reduced_synthetic = y_reduced_synthetic[selected_resampled == True]

    X_final = np.vstack((X_train, X_reduced_synthetic))
    y_final = np.hstack((y_train, y_reduced_synthetic))
    return X_final, y_final


def get_loss(loss_function, params, cls_num_list, device):
    try:
        if loss_function == LossFunction.BINARYVSLOSS:
            return BinaryVSLoss(iota_pos=params[0], iota_neg=params[1], Delta_pos=params[2], Delta_neg=params[3],
                                weight=[params[4], params[5]], device=device)
        if loss_function == LossFunction.VSLOSS:
            return VSLoss(cls_num_list, gamma=params[0], tau=params[1], weight=[params[2], params[3]], device=device)
        if loss_function == LossFunction.IBLOSS:
            return IBLoss(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3], device=device)
        if loss_function == LossFunction.IBFOCALLOSS:
            return IBFocalLoss(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3], gamma=params[4],
                               device=device)
        if loss_function == LossFunction.LDAMLOSS:
            return LDAMLoss(cls_num_list=cls_num_list, weight=[params[1], params[2]], max_m=params[0], s=params[3],
                            device=device)
        if loss_function == LossFunction.LDAMLOSS:
            return LDAMLoss(cls_num_list=cls_num_list, weight=[params[1], params[2]], max_m=params[0], s=params[3],
                            device=device)
        if loss_function == LossFunction.VSLOSSMDR:
            return VSLossMDR(cls_num_list, gamma=params[0], tau=params[1], weight=[params[2], params[3]], l=params[4],
                             device=device)
        if loss_function == LossFunction.LDAMLOSSMDR:
            return LDAMLossMDR(cls_num_list=cls_num_list, weight=[params[1], params[2]], max_m=params[0], s=params[3],
                               l=params[4],
                               device=device)
        if loss_function == LossFunction.IBLOSSMDR:
            return IBLossMDR(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3], l=params[4],
                             device=device)
        if loss_function == LossFunction.BINARYVSLOSSMDR:
            return BinaryVSLossMDR(iota_pos=params[0], iota_neg=params[1], Delta_pos=params[2], Delta_neg=params[3],
                                   weight=[params[4], params[5]], l=params[6], device=device)
        if loss_function == LossFunction.CROSSENTROPYLOSS:
            return CrossEntropyLoss(weight=[params[0], params[1]])
        #if loss_function == LossFunction.BOUNDEDEXPONENTIALLOSS:
        #    return BoundedExponentialLoss(eta=params[0], alpha=params[1])
        #if loss_function == LossFunction.COMPLEMENTCROSSENTROPYLOSS:
        #    return CCE(weight=[params[0], params[1]], balancing_factor=params[2])
    except Exception as e:
        print(e)
    return


def get_config_files(path):
    files = []
    dir_list = os.listdir(path)
    for file in dir_list:
        files.append(os.path.join(path, file))
    return files[0:WEAK_CLASSIFIERS_COUNT]


import pandas as pd


def load_keel_dat_file(file_path):
    data_started = False
    data_lines = []
    column_names = []

    # Open the .dat file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check if data section has started
            if data_started:
                data_lines.append(line)

            # Start reading data after @data marker
            if line.lower() == '@data':
                data_started = True

            # Collect column names from the header (before @data)
            elif line.startswith('@attribute'):
                column_names.append(line.split()[1])

    # Convert data lines into a DataFrame
    data = [row.split(',') for row in data_lines]
    df = pd.DataFrame(data, columns=column_names)
    return df


def get_classifier(clf_type, solution):
    if clf_type == Classifier.SVC:
        clf = SVC(random_state=RANDOM_STATE, gamma=solution[0], C=solution[1])
    if clf_type == Classifier.WeightedSVC:
        clf = SVC(random_state=RANDOM_STATE, gamma=solution[0], C=solution[1],
                  class_weight={0: solution[2], 1: solution[3]})
    if clf_type ==Classifier.BalancedCascade:
        criterion = 'gini'
        if solution[1]!=0:
            criterion='entropy'

        splitter = 'best'
        if solution[2]!=0:
            splitter='random'
        estimator = DecisionTreeClassifier(splitter=splitter, criterion=criterion, ccp_alpha=solution[3])
        clf = imb.BalanceCascadeClassifier(random_state=RANDOM_STATE, n_estimators=solution[0], estimator=estimator)
    if clf_type==Classifier.AdaCost:
        algorithm = 'SAMME'
        if solution[2] != 0:
            algorithm = 'SAMME.R'
        clf = imb.AdaCostClassifier(n_estimators=solution[0], learning_rate=solution[1],algorithm=algorithm)
    if clf_type==Classifier.SelfPaced:
        criterion = 'gini'
        if solution[1] != 0:
            criterion = 'entropy'
        splitter = 'best'
        if solution[2] != 0:
            splitter = 'entropy'
        clf = imb.SelfPacedEnsembleClassifier(estimator=DecisionTreeClassifier(criterion=criterion, splitter=splitter, ccp_alpha=solution[3]), random_state=RANDOM_STATE,n_estimators=solution[0])
    return clf
