import scipy.stats as statistic
import pandas as pd
from statsmodels.stats.multitest import multipletests


if __name__ == "__main__":
    # load results and extract GM scores
    # _real_data = pd.read_csv("./aggregated_real_data_gmean_scores.csv")
    # _synthetic_data = pd.read_csv("./aggregated_synthetic_data_gmean_scores.csv")

    # load results and extract AUC scores
    _real_data = pd.read_csv("./aggregated_real_data_auc_scores.csv")
    _synthetic_data = pd.read_csv("./aggregated_synthetic_data_auc_scores.csv")

    _joined_data = pd.concat([_real_data, _synthetic_data], axis=0, ignore_index=True)
    print(_joined_data)
    _joined_data = _joined_data.loc[:, ~_joined_data.columns.str.contains('Unnamed')]
    _pure_data = pd.DataFrame(_joined_data.iloc[:, 1:])
    _methods = _pure_data.columns

    _num_classifiers = _pure_data.shape[1]
    _p_values = []
    _comparison = []

    # calculating
    for _idx in range(0, (_num_classifiers - 1)):
        print(f"working on: {_methods[_idx]}")
        _stat, _p = statistic.wilcoxon(_pure_data.iloc[:, -1], _pure_data.iloc[:, _idx])
        _p_values.append(_p)
        _comparison.append(("GA-HESO", _methods[_idx]))

        print("########")
        print(f"{_stat}")
        print(f"{_p}")
        print("########")

    print(f"p_values: {_p_values}")
    print(f"comparison: {_comparison}")

    # Adjust p-values using False Discovery Rate (Benjamini-Hochberg method)
    _rejected, _p_adjusted, _, _ = multipletests(_p_values, alpha=0.05, method='fdr_bh')

    # Print results

    print(f"Original p-values: {_p_values}")
    print(f"Adjusted p-values (FDR): {_p_adjusted}")
    print(f"Significant Comparisons (FDR < 0.05): {[comp for comp, sig in zip(_comparison, _rejected) if sig]}")
