import pandas as pd

_gm_xgboost_scores = [60.7, 20.7, 65.8]
_auc_xgboost_scores = [63.5, 52.1, 70.0]


if __name__ == "__main__":
    # auc
    _df_auc = pd.read_csv("./aggregated_synthetic_data_auc_scores.csv")
    _df_auc = _df_auc.loc[:, ~_df_auc.columns.str.contains('Unnamed')]
    _df_auc["XGBOOST_SMOTE"] = _auc_xgboost_scores
    _df_auc.to_csv("./aggregated_synthetic_data_auc_scores.csv", index=False)

    # g-mean
    _df_gmean = pd.read_csv("./aggregated_synthetic_data_gmean_scores.csv")
    _df_gmean = _df_gmean.loc[:, ~_df_gmean.columns.str.contains('Unnamed')]
    _df_gmean["XGBOOST_SMOTE"] = _gm_xgboost_scores
    _df_gmean.to_csv("./aggregated_synthetic_data_gmean_scores.csv", index=False)
