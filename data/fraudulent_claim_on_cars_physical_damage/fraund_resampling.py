import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('training data.csv')

desired_samples = int(len(df) * 0.1)


df_subsampled = pd.concat([
    df[df['fraud'] == 0].sample(frac=desired_samples/len(df), random_state=42),
    df[df['fraud'] == 1].sample(frac=desired_samples/len(df), random_state=42)
])

print(df.shape)
print(df_subsampled.shape)

print(np.sum(df['fraud']))
print(np.sum(df_subsampled['fraud']))

#df_subsampled.to_csv('training data subsampled.csv', index=None)