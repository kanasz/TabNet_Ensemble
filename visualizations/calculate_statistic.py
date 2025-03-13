import math
import pandas as pd


if __name__ == "__main__":

    df_auc = pd.read_csv("./aggregated_real_data_auc_scores.csv")
    df_gmean = pd.read_csv("./aggregated_real_data_gmean_scores.csv")

    columns_auc = df_auc.columns
    columns_gmean = df_gmean.columns

    if len(columns_auc[2:]) == len(columns_gmean[2:]) and list(columns_auc[2:]) == list(columns_gmean[2:]):
        average_auc = df_auc.mean(numeric_only=True)
        std_auc = df_auc.std(numeric_only=True)
        average_gmean = df_gmean.mean(numeric_only=True)
        std_gmean = df_gmean.std(numeric_only=True)

        statistic_df = pd.DataFrame({
            'clf': columns_auc[2:],
            'average AUC': [math.ceil(value * 10) / 10 for value in average_auc[1:]],
            'std AUC': [math.ceil(value) for value in std_auc[1:]],
            'average GMEAN': [math.ceil(value * 10) / 10 for value in average_gmean[1:]],
            'std GMEAN': [math.ceil(value) for value in std_gmean[1:]]
        })

        statistic_df.to_csv("./averaged_auc_and_gmean_results.csv")
    else:
        raise Exception("Mismatch of extracted columns!")

    # extract datasets names
    dataset_names = df_auc.iloc[:, [1]]
    print(dataset_names)

    # perform ranking according to performance
    cleaned_auc_df = df_auc.iloc[:, 2:]
    cleaned_gmean_df = df_gmean.iloc[:, 2:]

    ranked_auc_df = cleaned_auc_df.apply(lambda row: row.rank(method='min', ascending=False), axis=1)
    ranked_gmean_df = cleaned_gmean_df.apply(lambda row: row.rank(method='min', ascending=False), axis=1)
    # convert ranking to integer format
    ranked_auc_df = pd.DataFrame(ranked_auc_df.astype(int))
    ranked_gmean_df = ranked_gmean_df.astype(int)

    concatenated_auc_df = pd.concat([dataset_names, ranked_auc_df], axis=1, ignore_index=False)
    concatenated_gmean_df = pd.concat([dataset_names, ranked_gmean_df], axis=1, ignore_index=False)

    average_ranking_auc = concatenated_auc_df.iloc[:, 1:].mean()
    average_ranking_gmean = concatenated_gmean_df.iloc[:, 1:].mean()
    print(f"Averages: {average_ranking_auc}")
    print(f"Averages rank GM: {average_ranking_gmean}")

    auc_median = concatenated_auc_df.iloc[:, 1:].median()
    gmean_median = concatenated_gmean_df.iloc[:, 1:].median()
    print(f"Median AUC score: {auc_median}")
    print(f"Median GMEAN score: {gmean_median}")

    winning_times_auc = (concatenated_auc_df.iloc[:, 1:] == 1).sum()
    winning_times_gmean = (concatenated_gmean_df.iloc[:, 1:] == 1).sum()

    print(f"Winning times for AUC score:\n {winning_times_auc}")
    print(f"Winning times for AUC score:\n {winning_times_gmean}")

    concatenated_auc_df.to_csv("./ranked_auc_results.csv")
