import time
import pandas as pd

def stream_engine_data(df, engine_id):
    """Simulates a live IoT sensor feed."""
    engine_data = df[df['unit_nr'] == engine_id]
    for _, row in engine_data.iterrows():
        # In a real app, this would be sent to a WebSocket or Database
        yield row
        time.sleep(0.5) # Simulate 2Hz sensor frequency