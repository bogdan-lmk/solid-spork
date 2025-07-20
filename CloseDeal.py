import onnx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from onnx import shape_inference
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier

# Загрузка данных
df = pd.read_csv("dataset_opt.csv")

# Отделяем признаки и целевую переменную
X = df.drop(columns=["target"])
y = df["target"].astype(int)

# Стандартизация признаков
X = pd.get_dummies(X)  # на случай наличия категориальных переменных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели XGBoost
#model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
#model = XGBClassifier(
#    n_estimators=100,
#    max_depth=5,
#    learning_rate=0.1,
#    scale_pos_weight=3,  # 💡 Добавлено для компенсации дисбаланса
#    random_state=42
#)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

importance = model.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=X.columns)
plt.title("XGBoost Feature Importance")
plt.show()

# Оценка модели
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("📈 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Average Precision Score:", average_precision_score(y_test, y_prob))

# Сохраним .pkl
joblib.dump(model, "short_model.pkl")
# Конвертация в ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
#onnx_model = convert_xgboost(model, initial_types=initial_type)
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("short_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model saved as closing_model.onnx")
