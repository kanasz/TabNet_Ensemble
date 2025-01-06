import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from imblearn.datasets import make_imbalance

tabnet_max_epochs = 300


def plot_tsne(X, y):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    X = tsne.fit_transform(X).reshape(len(X), 2)
    df = pd.DataFrame(X)
    df.columns = ["comp-1", "comp-2"]
    df['y'] = y
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), style=df.y, markers=["P", "o"],
                    data=df).set(title=" ")
    plt.show()


def generate_imbalanced_dataset(ratio, size, n_informative, n_redundant,n_features, n_clusters_per_class):
    # Generate an initial dataset larger than needed, so we have samples to downsample from
    initial_size = int(size * (1 + ratio) / min(1, ratio / (1 + ratio)))  # Adjust initial size to ensure enough samples
    X, y = make_classification(n_classes=2, class_sep=2,
                               weights=[0.5, 0.5], n_informative=n_informative,
                               n_redundant=n_redundant, flip_y=0, n_features=n_features,
                               n_clusters_per_class=n_clusters_per_class, n_samples=initial_size, random_state=10)

    # Define the sampling strategy based on the desired ratio
    minor_class = max(5, int(size / (1 + ratio))) # Ensure at least 5 samples in the minority class
    majority_class = size - minor_class  # Remaining samples for majority class

    # Apply imbalance using downsampling
    X_imb, y_imb = make_imbalance(X, y, sampling_strategy={0: majority_class, 1: minor_class}, random_state=10)
    return X_imb, y_imb



if __name__ == "__main__":

    contaminations = [2, 5,10,20, 50]
    '''
    print("GENERATING 250 SAMPLES DATASETS")
    for contamination in contaminations:
        n_features = 20
        n_informative = 10
        n_train = 250
        n_clusters_per_class = 10
        n_redundant = 5
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '01_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    for contamination in contaminations:
        n_features = 200
        n_informative = 30
        n_train = 250
        n_clusters_per_class = 40
        n_redundant = 50
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '02_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    print("GENERATING 500 SAMPLES DATASETS")
    for contamination in contaminations:
        n_features = 20
        n_informative = 7
        n_train = 500
        n_clusters_per_class = 20
        n_redundant = 5
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '03_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    for contamination in contaminations:
        n_features = 200
        n_informative = 70
        n_train = 500
        n_clusters_per_class = 15
        n_redundant = 40
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '04_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    print("GENERATING 300 SAMPLES DATASETS")
    for contamination in contaminations:
        n_features = 100
        n_informative = 50
        n_train = 300
        n_clusters_per_class = 8
        n_redundant = 20
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '05_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    for contamination in contaminations:
        n_features = 100
        n_informative = 30
        n_train = 300
        n_clusters_per_class = 10
        n_redundant = 50
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '06_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))
    '''
    for contamination in contaminations:
        n_features = 100
        n_informative = 70
        n_train = 300
        n_clusters_per_class = 10
        n_redundant = 10
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '07_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))

    for contamination in contaminations:
        n_features = 50
        n_informative = 30
        n_train = 300
        n_clusters_per_class = 10
        n_redundant = 10
        n_classes = 2
        X, y = generate_imbalanced_dataset(contamination, n_train, n_informative, n_redundant, n_features,n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '08_sensitivity_synthetic_imb_{}_features_{}_samples_{}.csv'.format(str(1/contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))
