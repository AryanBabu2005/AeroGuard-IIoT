import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_heatmap(df):
    """
    Shows which sensors are most 'statistically significant' 
    for predicting Remaining Useful Life (RUL).
    """
    # Select only numeric sensor columns and RUL
    cols_to_corr = [col for col in df.columns if 's_' in col or col == 'RUL']
    corr_matrix = df[cols_to_corr].corr()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix[['RUL']].sort_values(by='RUL', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Sensor Correlation with Remaining Useful Life (RUL)")
    plt.show()

def plot_sensor_trend(df, unit_nr, sensor_name):
    """
    Plots a specific engine's sensor trend over time.
    High-level: Shows how noise is reduced by the rolling mean.
    """
    engine_data = df[df['unit_nr'] == unit_nr]
    
    plt.figure(figsize=(12, 5))
    plt.plot(engine_data['time_cycles'], engine_data[sensor_name], alpha=0.3, label='Raw Sensor')
    plt.plot(engine_data['time_cycles'], engine_data[f"{sensor_name}_mean"], color='red', label='Rolling Mean (Trend)')
    
    plt.title(f"Degradation Trend for Engine #{unit_nr} - {sensor_name}")
    plt.xlabel("Time Cycles")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.show()

def analyze_lifespan_distribution(df):
    """
    Calculates the Probability Density Function (PDF) of engine failures.
    Essential for the Statistics portion of the project.
    """
    lifespans = df.groupby('unit_nr')['time_cycles'].max()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(lifespans, kde=True, color='green', bins=15)
    plt.title("Distribution of Engine Lifespans (Probability Density)")
    plt.xlabel("Total Cycles until Failure")
    plt.ylabel("Frequency")
    plt.show()
    
    print(f"--- Statistical Summary ---")
    print(f"Mean Time to Failure (MTTF): {lifespans.mean():.2f} cycles")
    print(f"Standard Deviation: {lifespans.std():.2f}")