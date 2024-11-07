import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE

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


if __name__ == "__main__":

    contaminations = [1/2, 1/5,1/10,1/20,1/50]

    print("GENERATING 250 SAMPLES DATASETS")
    for contamination in contaminations:
        n_features = 20
        n_informative = 10
        n_train = 250
        n_clusters_per_class = 10
        n_redundant = 10
        n_classes = 2
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '01_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
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
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '02_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
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
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '03_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
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
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '04_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
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
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '05_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
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
        X, y = make_classification(n_samples=n_train, n_features=n_features, n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=n_classes, random_state=12, weights=[1 - contamination],
                                   n_clusters_per_class=n_clusters_per_class)
        columns = [f"feature{i + 1}" for i in range(n_features)]
        df = pd.concat([pd.DataFrame(X, columns=columns), pd.Series(y, name='target')], axis=1)
        df.to_csv(
            '06_sensitivity_synthetic_{}_contamination.csv'.format(str(contamination),
                                                                                                  str(n_features), n_train),
            index=None)
        #plot_tsne(X, y)
        print(np.sum(y))
