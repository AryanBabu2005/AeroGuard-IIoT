from src.data_loader import load_data
from src.feature_eng import add_rolling_features
from src.modeling import train_predictive_model
from src.visualization import plot_predictions, plot_error_distribution
from sklearn.model_selection import train_test_split

# 1. Load & Engineer
train_df = load_data('train_FD001.txt')
train_pro = add_rolling_features(train_df)

# 2. Train
model, features = train_predictive_model(train_pro)

# 3. Create a validation set to visualize performance
X = train_pro[features]
y = train_pro['RUL']
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Predict on validation set
y_pred = model.predict(X_val)

# 5. Visualize (The 'Show and Tell' for your Professor)
# We plot the first 100 samples so the graph isn't too crowded
plot_predictions(y_val.values[:100], y_pred[:100])
plot_error_distribution(y_val.values, y_pred)