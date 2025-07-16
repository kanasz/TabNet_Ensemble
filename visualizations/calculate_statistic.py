import math
import pandas as pd


# calculate necessary statistic for selected .csv file, produced by "analyze_output.py" script
def __calculate_average_results(path_to_csv_file, metric, path_to_save):
    _results_data = pd.read_csv(path_to_csv_file)
    # if needed
    _results_data.fillna(0, inplace=True)

    _columns = _results_data.columns
    _mean_values = _results_data.mean(numeric_only=True)
    _std_values = _results_data.std(numeric_only=True)

    _output = pd.DataFrame({
        'clf': _columns[2:],
        f'average {metric}': [math.ceil(value * 10) / 10 for value in _mean_values[1:]],
        f'std {metric}': [math.ceil(value) for value in _std_values[1:]]
    })

    _output.to_csv(f'./analyze_output/{path_to_save}')

    _dataset_names = _results_data.iloc[:, [1]]
    # print(_dataset_names)
    _df_with_only_results = _results_data.iloc[:, 2:]
    _win_rank_df = _df_with_only_results.apply(lambda row: row.rank(method='min', ascending=False), axis=1).astype(int)
    _output_of_win_statistic = pd.concat([_dataset_names, _win_rank_df], axis=1, ignore_index=False)

    # calculate average of winning ranking
    print("################")
    print("Averaged winning ranking:\n")
    print(_output_of_win_statistic.iloc[:, 1:].mean())
    print("################")

    # calculate median of winning ranking
    print("Median for winning ranking:\n")
    print(_output_of_win_statistic.iloc[:, 1:].median())
    print("################")

    # sum number of wins per method
    print("Number of wins per method:\n")
    print((_output_of_win_statistic.iloc[:, 1:] == 1).sum())
    print("################")


if __name__ == "__main__":

    """
    __calculate_average_results(path_to_csv_file="analyze_output/aggregated_real_data_auc_scores.csv",
                                metric='auc',
                                path_to_save="averaged_real_data_auc.csv")

    __calculate_average_results(path_to_csv_file="aggregated_real_data_gmean_scores.csv",
                                metric='gmean',
                                path_to_save="averaged_real_data_gmean.csv")
    """
    __calculate_average_results(path_to_csv_file="aggregated_ablation_real_data_auc_scores.csv",
                                metric='auc',
                                path_to_save="averaged_ablation_real_data_auc.csv")

    """
    __calculate_average_results(path_to_csv_file="aggregated_ablation_real_data_gmean_scores.csv",
                                metric='gmean',
                                path_to_save="averaged_ablation_real_data_gmean.csv")
    """
