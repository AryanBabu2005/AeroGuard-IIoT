import pandas as pd
import os

def load_data(file_name):
    # Path handling as we discussed
    data_path = os.path.join('CMAPSSData', file_name)
    
    # NASA Column Names
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    
    df = pd.read_csv(data_path, sep='\s+', header=None, names=col_names)
    
    # Calculate RUL immediately
    # Change max to "max"
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform("max") - df['time_cycles']
    return df