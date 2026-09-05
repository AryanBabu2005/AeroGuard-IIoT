import numpy as np

def create_sequences(data, window_size=20):
    """
    Converts 2D sensor data into 3D sequences.
    From: (Rows, Sensors) -> To: (Samples, 20, 21)
    """
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
    return np.array(X)