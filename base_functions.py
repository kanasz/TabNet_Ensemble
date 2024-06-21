import csv
import os
from pathlib import Path
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score


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
    # features = pd.get_dummies(features, columns=['gender','living_status','accident_site','channel','vehicle_category','vehicle_color'])
    labels = data['fraud']
    return features, labels

def get_aids_data(features):
    path = Path(__file__).parent / "data/aids_classification/aids_classification_{}.csv".format(features)
    data = pd.read_csv(path)
    data = data.dropna()
    features = data.drop(["infected"], axis=1)
    labels = data['infected']
    return features, labels
