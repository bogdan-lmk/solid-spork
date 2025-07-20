import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Загрузка данных
df = pd.read_csv('data.csv')

# 2. Разделение признаков и целевой переменной
X = df.drop(columns='target')
y = df['target']

# 3. Разделение на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Обучение модели
#model = RandomForestClassifier(random_state=42, class_weight='balanced')
model = XGBClassifier(scale_pos_weight=8, use_label_encoder=False, eval_metric='logloss')

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
model.fit(X_train_res, y_train_res)

importance = model.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=X.columns)
plt.title("XGBoost Feature Importance")
plt.show()

# 5. Оценка модели
y_pred = model.predict(X_test)
print("Классификационный отчёт:")
print(classification_report(y_test, y_pred))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
probs = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, probs))
print("Average Precision Score:", average_precision_score(y_test, probs))
# Предсказания по пользовательскому порогу
y_pred_custom = (probs > 0.28)
print("\n📊 Отчёт при кастомном пороге:")
print(classification_report(y_test, y_pred_custom))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_custom))

probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Найди F1 для каждого порога
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[f1_scores.argmax()]
print("Best threshold:", best_threshold)

# 6. Сохранение модели
joblib.dump(model, 'trained_model.pkl')
print("Модель сохранена в 'trained_model.pkl'")
# Загружаем модель
model = joblib.load("trained_model.pkl")

# Пример входных данных (размерность)
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# Конвертация
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Сохраняем
with open("trained_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
