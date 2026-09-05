import time

def get_live_sensor_stream(df, engine_id):
    """
    Generator function to simulate real-time IoT data streaming.
    Yields one 'packet' of sensor data every second.
    """
    engine_data = df[df['unit_nr'] == engine_id].copy()
    
    for i in range(len(engine_data)):
        # Simulate the 'Current' state of the engine
        yield engine_data.iloc[i]
        time.sleep(0.8) # Adjust for 'Real-time' feel