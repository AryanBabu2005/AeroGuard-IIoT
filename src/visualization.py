import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred):
    """
    Plots Predicted RUL vs Actual RUL to show model accuracy.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual RUL', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted RUL', color='red', linestyle='--', linewidth=2)
    plt.title('Comparison: Actual vs Predicted Remaining Useful Life')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Cycles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_error_distribution(y_true, y_pred):
    """
    Plots the distribution of prediction errors (Residuals).
    A 'Pro' statistics move to show if the model is biased.
    """
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='purple')
    plt.title('Prediction Error Distribution (Residuals)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.show()