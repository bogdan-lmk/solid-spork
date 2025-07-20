import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the processed dataset with engineered features and target
df = pd.read_csv("dataset_with_engineered_features.csv")

# 2. Select test set as the final 20% of the time series
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].reset_index(drop=True)

# 3. Prepare feature matrix and true target values
feature_cols = [
    'atr','atr_stop','atr_to_price_ratio','fast_ema','slow_ema','ema_fast_deviation',
    'pchange','avpchange','gma','positionBetweenBands','choppiness_index',
    'volatility_percent','rsi_volatility','adx','rsi_delta','linear_regression',
    'close_diff','normalized_price_change','rolling_mean_close','rolling_std_close'
]
X_test = test_df[feature_cols].astype(float)
# Rename feature columns to match model training
X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]
true_close = test_df['close_price_next'].values
base_close = test_df['close'].values

# 4. Load the trained model (update filename if different)
model = joblib.load("xgb_model.pkl")

# Align X_test to model's expected features
expected_feats = model.feature_names_
for feat in expected_feats:
    if feat not in X_test.columns:
        X_test[feat] = 0
# Reorder columns
X_test = X_test[expected_feats]

# 5. Predict returns and reconstruct absolute prices
predicted_delta = model.predict(X_test)
predicted_close = base_close * (1 + predicted_delta)

# 6. Compute and print evaluation metrics
mae = mean_absolute_error(true_close, predicted_close)
mse = mean_squared_error(true_close, predicted_close)
r2  = r2_score(true_close, predicted_close)
max_err = abs(true_close - predicted_close).max()

print(f"Test set size: {len(test_df)} samples")
print(f"Max Error: {max_err:.2f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# Plot for actual price
plt.figure(figsize=(12, 5))
plt.scatter(X_test.index, true_close, label="Actual close_price_next", marker='o', s = 20)
plt.title("Actual close_price_next price")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for predicted price
plt.figure(figsize=(12, 5))
plt.scatter(X_test.index, predicted_close, label="Model predicted price", marker='x',  s = 20)
plt.title("Model predicted close_price_next price")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()