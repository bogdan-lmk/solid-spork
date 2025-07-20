import numpy as np
import pandas as pd
from onnxmltools import convert_xgboost
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import joblib
import optuna
from catboost import CatBoostRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

def transform(df, bollinger_window=20):
    """
    Compute only the missing columns for the input DataFrame, assuming all other
    engineered features are already present in df.
    Only calculates:
      - close_diff
      - normalized_price_change
      - rolling_mean_close
      - rolling_std_close
      - close_price_next
      - target_return
    """
    # Convert existing feature columns to numeric, replacing comma decimals and filling NaNs with 0
    input_cols = [
        'atr','atr_stop','atr_to_price_ratio','fast_ema','slow_ema','ema_fast_deviation',
        'pchange','avpchange','gma','positionBetweenBands','choppiness_index',
        'volatility_percent','rsi_volatility','adx','rsi_delta','linear_regression'
    ]
    for col in input_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0)

    df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df['close_diff'] = df['close'].diff()
    df['normalized_price_change'] = df['close_diff'] / df['close'].shift(1)
    df['rolling_mean_close'] = df['close'].rolling(bollinger_window).mean()
    df['rolling_std_close'] = df['close'].rolling(bollinger_window).std()
    df['close_price_next'] = df['close'].shift(-1)
    df['target_return'] = df['close_price_next'] / df['close'] - 1
    # Remove last row (contains zeros in last column)
    df = df.iloc[:-1]
    return df.reset_index(drop=True)

# ========= Optuna hyperparameter optimization ==========
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_seed": 42,
        "verbose": False
    }

    cv = TimeSeriesSplit(n_splits=3)
    maes = []

    for train_idx, valid_idx in cv.split(X_filtered):
        X_train, X_valid = X_filtered.iloc[train_idx], X_filtered.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        predicted_close = df["close"].iloc[valid_idx].astype(float) * (1 + y_pred)
        true_close = df["close_price_next"].iloc[valid_idx].astype(float)

        mae = mean_absolute_error(true_close, predicted_close)
        maes.append(mae)

    return np.mean(maes)


raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
processed_df = transform(raw_df)

required_cols = [
        'atr','atr_stop','atr_to_price_ratio','fast_ema','slow_ema','ema_fast_deviation',
        'pchange','avpchange','gma','positionBetweenBands','choppiness_index',
        'volatility_percent','rsi_volatility','adx','rsi_delta','linear_regression',
        'close_diff','normalized_price_change','rolling_mean_close','rolling_std_close',
        'close','close_price_next','target_return'
    ]
processed_df = processed_df[required_cols]

processed_df.to_csv("dataset_with_engineered_features.csv", index=False)

# Load data
df = pd.read_csv("dataset_with_engineered_features.csv", decimal=",")

features = df.columns.drop(["close", "close_price_next", "target_return"])
X = df[features]
y = df["target_return"]
y_true = df["close_price_next"].to_numpy()
# Rename columns X -> f0, f1, ...
X.columns = [f"f{i}" for i in range(X.shape[1])]
# Convert all features to float
X = X.astype("float32")
# Training

# Split dataset

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    y_pred = model.predict(X_test)

    predicted_close = df["close"].iloc[test_index].astype(float) * (1 + y_pred)
    true_close = df["close_price_next"].iloc[test_index].astype(float)

    mae = mean_absolute_error(true_close, predicted_close)
    mse = mean_squared_error(true_close, predicted_close)
    r2 = r2_score(true_close, predicted_close)

    print("Fold Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}\n")

# Model
model = CatBoostRegressor(
    iterations=1999,
    learning_rate=0.004143405793827955,
    depth=5,
    l2_leaf_reg=0.01426311573116693,
    bagging_temperature=0.2467198633533077,
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Prediction
y_pred = model.predict(X_test)
df = df.astype("float32")
# Restore absolute values of close_price_next from predicted delta
predicted_close = df["close"].iloc[X_test.index].astype(float) * (1 + y_pred)
true_close = df["close_price_next"].iloc[X_test.index].astype(float)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()

# Recreate X from df with original names
X_original = df.drop(columns=["close_price_next"]).copy()

# Retrain model on original features
model.fit(X_original, y)

# Select important features by real names
importances = pd.Series(model.feature_importances_, index=X_original.columns)
important_features = importances[importances > 0.01].index.tolist()

# Filter by selected features
X_filtered = X_original[important_features].copy()

# Rename for ONNX compatibility
X_filtered.columns = [f"f{i}" for i in range(X_filtered.shape[1])]

# To run optimization, uncomment the following lines:
#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=50)
#print(f"Best parameters: {study.best_params}")

# Further retraining:
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
# Recalculate predicted_close after retraining
y_pred = model.predict(X_test)
# Restore absolute close_price_next from delta
predicted_close = df["close"].iloc[X_test.index].astype(float) * (1 + y_pred)
true_close = df["close_price_next"].iloc[X_test.index].astype(float)

# Metrics on prices
mae = mean_absolute_error(true_close, predicted_close)
mse = mean_squared_error(true_close, predicted_close)
r2 = r2_score(true_close, predicted_close)

# Deviations
abs_errors = np.abs(true_close - predicted_close)
print(f"🔺 Max Error: {np.max(abs_errors):.2f}")
print(f"📉 Mean Error: {np.mean(abs_errors):.2f}")

print("📊 Metrics (absolute price prediction):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")


# Create and train ensemble of models
estimators = [
    ('catboost', CatBoostRegressor(iterations=1927, learning_rate=0.004143405793827955, depth=5, random_seed=42, verbose=0)),
    ('xgboost', XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)),
    ('lightgbm', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42))
]

#stacking_regressor = StackingRegressor(
#    estimators=estimators,
#    final_estimator=CatBoostRegressor(iterations=500, learning_rate=0.01, depth=4, random_seed=42, verbose=0)
#)

#stacking_regressor.fit(X_train, y_train)

# Prediction and evaluation
#stack_pred = stacking_regressor.predict(X_test)

#predicted_close_stack = df["close"].iloc[X_test.index].astype(float) * (1 + stack_pred)
#true_close_stack = df["close_price_next"].iloc[X_test.index].astype(float)

#mae_stack = mean_absolute_error(true_close_stack, predicted_close_stack)
#mse_stack = mean_squared_error(true_close_stack, predicted_close_stack)
#r2_stack = r2_score(true_close_stack, predicted_close_stack)

#print("Ensemble model metrics:")
#print(f"MAE: {mae_stack:.4f}")
#print(f"MSE: {mse_stack:.4f}")
#print(f"R²: {r2_stack:.4f}")

# Save model to .pkl
joblib.dump(model, "xgb_model.pkl")

# Export to ONNX
#initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
#onnx_model = convert_xgboost(model, initial_types=initial_type)

#with open("xgb_model.onnx", "wb") as f:
#    f.write(onnx_model.SerializeToString())

#print("✅ Model exported to xgb_model.onnx and xgb_model.pkl")
