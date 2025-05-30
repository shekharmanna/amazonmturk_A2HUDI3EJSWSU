import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Simulate synthetic sensor data
def simulate_data(n_vehicles=100, n_timesteps=50):
    features = ['temperature', 'vibration', 'oil_pressure']
    data = []
    for vehicle_id in range(n_vehicles):
        base_rul = np.random.randint(80, 200)
        for t in range(n_timesteps):
            remaining_rul = base_rul - t
            if remaining_rul <= 0:
                break
            data.append({
                'vehicle_id': vehicle_id,
                'time_step': t,
                'temperature': 70 + 10 * np.random.randn(),
                'vibration': 0.5 + 0.1 * np.random.randn(),
                'oil_pressure': 30 + 5 * np.random.randn(),
                'RUL': remaining_rul
            })
    return pd.DataFrame(data)

# 2. Feature engineering using rolling window
def extract_features(group, features, window=5):
    df_feat = group.sort_values('time_step').copy()
    for col in features:
        df_feat[f'{col}_mean'] = df_feat[col].rolling(window, min_periods=1).mean()
        df_feat[f'{col}_std'] = df_feat[col].rolling(window, min_periods=1).std().fillna(0)
    return df_feat

# 3. Prepare training dataset
def prepare_data(df):
    features = ['temperature', 'vibration', 'oil_pressure']
    df_proc = df.groupby('vehicle_id').apply(lambda g: extract_features(g, features)).reset_index(drop=True)
    latest = df_proc.groupby('vehicle_id').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    feature_cols = [c for c in latest.columns if any(k in c for k in ['mean', 'std'])]
    return latest[feature_cols], latest['RUL']

# 4. Train model
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# 5. Verifier
def test_rul_predictions(model, X_test, y_test, tolerance=10):
    y_pred = model.predict(X_test)
    error = np.abs(y_pred - y_test)
    accuracy = (error < tolerance).mean()
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Prediction Accuracy (within ±{tolerance} RUL): {accuracy:.2%}")
    assert accuracy > 0.85, "Prediction accuracy below expected threshold"

# --- Run full pipeline ---
if __name__ == "__main__":
    np.random.seed(42)
    df = simulate_data()
    X, y = prepare_data(df)
    model, X_test, y_test = train_model(X, y)
    test_rul_predictions(model, X_test, y_test)
