import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from base_functions import get_abalone_20_vs_8_9_10_data, get_abalone_19_vs_10_11_12_13_data, get_abalone_9_vs_18_data, \
    get_abalone_3_vs_11_data, get_meanshift_cluster_counts

data = get_abalone_20_vs_8_9_10_data()
data = get_abalone_19_vs_10_11_12_13_data()
data = get_abalone_9_vs_18_data()
data = get_abalone_3_vs_11_data()


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def generate_folds(X,y, numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('scaler', StandardScaler())  # Standardize features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values in categorical features
        ('onehot', OneHotEncoder())  # OneHot encode categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    def create_meanshift_pipeline(bandwidth=None):
        return Pipeline(steps=[
            ('meanshift', MeanShift(bandwidth=bandwidth))  # Apply MeanShift clustering
        ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Step 5: Preprocess training data (imputation, scaling, one-hot encoding)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Step 6: Apply SMOTE to oversample the minority class in the training set
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_preprocessed, y_train)

        # Step 7: Separate the synthetic samples for the minority class
        n_generated_samples = len(X_resampled) - len(X_train_preprocessed)  # Number of synthetic samples
        synthetic_samples = X_resampled[-n_generated_samples:]  # Synthetic samples are at the end
        synthetic_labels = y_resampled[-n_generated_samples:]  # Corresponding labels for synthetic samples

        # Step 8: Cluster only the synthetic samples
        bandwidth = estimate_bandwidth(synthetic_samples, quantile=0.05)
        clustering_pipeline = create_meanshift_pipeline(bandwidth)
        clustering_pipeline.fit(synthetic_samples)

        cluster_centers = clustering_pipeline.named_steps['meanshift'].cluster_centers_
        y_reduced_synthetic = np.full(shape=cluster_centers.shape[0], fill_value=1)
        final = np.vstack((X_train_preprocessed, cluster_centers))
        y_final = np.hstack((y_train, y_reduced_synthetic))
        print(len(cluster_centers))



numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
categorical_features = ['Sex']
print('get_abalone_9_vs_18_data')
X, y = get_abalone_9_vs_18_data()

clusters = get_meanshift_cluster_counts(X, y, numeric_features, categorical_features)
print(clusters)

generate_folds(X, y, numeric_features, categorical_features)
'''
print('get_abalone_20_vs_8_9_10_data')
X, y = get_abalone_20_vs_8_9_10_data()
generate_folds(X, y, numeric_features, categorical_features)
print('get_abalone_19_vs_10_11_12_13_data')
X, y = get_abalone_19_vs_10_11_12_13_data()
generate_folds(X, y, numeric_features, categorical_features)
print('get_abalone_3_vs_11_data')
X, y = get_abalone_3_vs_11_data()
generate_folds(X, y, numeric_features, categorical_features)
'''

    # X_final = np.vstack((X_train, X_reduced_synthetic))
    # y_final = np.hstack((y_train, y_reduced_synthetic))
