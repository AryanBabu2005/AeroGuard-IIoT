from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def get_rf_model():
    return RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

def get_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear') # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model