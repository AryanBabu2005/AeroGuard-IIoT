import pandas as pd
def add_rolling_features(df, window_size=10):
    active_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    
    # Rolling Mean
    df_mean = df.groupby('unit_nr')[active_sensors].rolling(window=window_size).mean().reset_index(level=0, drop=True)
    df_mean.columns = [col + '_mean' for col in df_mean.columns]
    
    # Rolling Std
    df_std = df.groupby('unit_nr')[active_sensors].rolling(window=window_size).std().reset_index(level=0, drop=True)
    df_std.columns = [col + '_std' for col in df_std.columns]
    
    return pd.concat([df, df_mean, df_std], axis=1).dropna()