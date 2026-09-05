import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Fixed the typo here!

def train_predictive_model(df):
    """
    Trains a high-level regression model to predict Remaining Useful Life (RUL).
    """
    # 1. Feature Selection: Use the engineered features
    features = [col for col in df.columns if '_mean' in col or '_std' in col]
    X = df[features]
    y = df['RUL']

    # 2. Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Initialization
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    print("Training the model... please wait.")
    model.fit(X_train, y_train)

    # 4. Predictions & Evaluation
    y_pred = model.predict(X_test)
    
    # RMSE calculation (updated for newer sklearn versions)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5 
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Training Complete.")
    print(f"📊 Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
    print(f"📈 R2 Score (Accuracy): {r2:.2f}")

    # Save the model professionally
    joblib.dump(model, 'engine_model.pkl')
    print("💾 Model saved as 'engine_model.pkl'")

    return model, features