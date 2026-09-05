from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential([
        # Layer 1: Captures temporal trends from the 20-cycle window
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), 
        
        # Layer 2: Final feature extraction
        LSTM(units=30, return_sequences=False),
        Dropout(0.2),
        
        # Output: Predicting the continuous RUL value (Regression)
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model