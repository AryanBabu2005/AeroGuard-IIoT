import pandas as pd
import numpy as np

def clean_data(df):
    # Drop columns with 0 variance (sensors that don't change)
    drop_cols = ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    return df.drop(columns=drop_cols, errors='ignore')

def add_rolling_features(df, window=20):
    feature_cols = [c for c in df.columns if c.startswith('s_')]
    for col in feature_cols:
        df[f'{col}_mean'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window, 1).mean())
        df[f'{col}_std'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window, 1).std().fillna(0))
    return df

def create_sequences(data, feature_cols, window=20):
    """Specific for LSTM: Converts flat data into 3D blocks"""
    X = []
    for i in range(len(data) - window + 1):
        X.append(data.iloc[i : i + window][feature_cols].values)
    return np.array(X)