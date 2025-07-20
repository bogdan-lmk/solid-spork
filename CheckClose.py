import pandas as pd
from onnxmltools import convert_xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
import joblib
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Загружаем данные
df = pd.read_csv("binary_classification_dataset.csv")  # Заменить на актуальный путь

# Отбираем признаки и целевую переменную
features = [
    "choppiness_index",
    "volatility_percent",
    "linear_regression",
    "atr_to_price_ratio",
    "rsi_delta",
    "rsi_volatility",
    "bollinger_position",
    "gma"
]
X = df[features]
y = df["target"]

# Стандартизация признаков
X = pd.get_dummies(X)  # на случай наличия категориальных переменных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Обучение модели
model = XGBClassifier(scale_pos_weight=1.5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Предсказания и метрики
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("📈 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Average Precision Score:", average_precision_score(y_test, y_prob))


# Сохраним .pkl
joblib.dump(model, "close_position_model.pkl")
# Конвертация в ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open("close_position_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model saved as closing_model.onnx")
