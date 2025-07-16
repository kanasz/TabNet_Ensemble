import pandas as pd

df = pd.read_csv("SMOTE_MS_synthetic_results_by_folds.csv")
#df = pd.read_csv("ADASYN_MS_results_by_folds.csv")
#df = pd.read_csv("SMOTE_DBSCAN_results_by_folds.csv")
agg_df = df.groupby([ 'name']).mean().reset_index()
print(agg_df)

agg_df.to_csv("synthetic_aggregated.csv",index=None)

df = pd.read_csv("SMOTE_MS_results_by_folds.csv")
#df = pd.read_csv("ADASYN_MS_results_by_folds.csv")
#df = pd.read_csv("SMOTE_DBSCAN_results_by_folds.csv")
agg_df = df.groupby([ 'name']).mean().reset_index()
print(agg_df)

agg_df.to_csv("real_aggregated.csv",index=None)