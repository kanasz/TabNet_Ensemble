import numpy as np
import pandas as pd
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
import os

#OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_ss_06_imb_0.2_feat_100_samples_300

imb_ratios = ['0.5', '0.2', '0.1', '0.05', '0.02']
datasets = [('01','20','250'), ('02','200','250'), ('03','20','500'), ('04','200','500'), ('05','100','300'), ('06','100','300')
            , ('07','100','300'), ('08','50','300')]


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
    #print("\t{}".format(np.average(gmeans)) )
    #print("gmean: {:.2f}±{}, auc: {:.2f}±{}".format(np.average(gmeans)*100,int(round(np.std(gmeans)*100)),
    #                                                np.average(roc_aucs)*100,int(round(np.std(roc_aucs)*100))))

    return "{}".format(np.average(gmeans))

df = pd.DataFrame(columns=['dataset'] + imb_ratios)

for ds in datasets:
    id, features, samples = ds
    dataset = "{}_{}".format(features, samples)
    row = {
        "dataset": [dataset]
    }
    for imb in imb_ratios:


        file_path = "predictions/synthetic_sensitivity/results/UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_ss_{}_imb_{}_feat_{}_samples_{}.txt".format(id,imb,features,samples)
        if os.path.exists(file_path):
            row[imb] = [process_results(file_path)]
            print(file_path)
    new_row = pd.DataFrame(row)
    df = pd.concat([df, new_row], ignore_index=True)
print(df)
df.to_csv("aggregated_synthetic.csv", index=None)