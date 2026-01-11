import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("diamonds_clean_final.csv")

# Используем логарифм цены
df['log_price'] = np.log10(df['price'].clip(lower=1))

# квантильное разбиение
df['price_segment'] = pd.qcut(
    df['log_price'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

print(df['price_segment'].value_counts(normalize=True))

# Целевая переменная
y = df['price_segment']

# Признаки
features = [
    'stone.carat',
    'stone.depth',
    'stone.tableSize',
    'enriched.ratio',
    'stone.color.name',
    'stone.clarity.name',
    'stone.cut.name',
    'stone.shape.name'
]

X = df[features]




# Проверяем есть ли NaN в х и у
print("NaN in y:", y.isna().sum())
print(y.value_counts(dropna=False))

print("NaN in X:", X.isna().sum().sort_values(ascending=False).head(20))

# Убираем строки, где нет целевого класса
mask = y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
print("NaN in y after cleaning:", y.isna().sum()) # Проверяем, что эти строки удалились

# Проверяем распределение классов после очистки
print(y.value_counts(normalize=True))


# Кодирование категориальных признаков
cat_features = [
    'stone.color.name',
    'stone.clarity.name',
    'stone.cut.name',
    'stone.shape.name'
]

num_features = [
    'stone.carat',
    'stone.depth',
    'stone.tableSize',
    'enriched.ratio',
]



# Заменяем пропуски (NaN) медианой этого признака
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Заменяем пропуски в категориальных признаках категорией "Unknown" с последующим One-Hot кодированием
# Это позволяет сохранить информацию об отсутствии данных и избежать искажения распределений
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# результаты объединяются и передаются в модель классификации
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)


# Train / Test split (Разделение набора на обучающую и тестовую выборку)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y     #Сохраняет одинаковое распределение классов в обучающей и тестовой выборках
)

# Decision Tree (Дерево решений)
model_dt = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

model_dt.fit(X_train, y_train)


# Проверяем, что пропуски в stone.cut.name были заменены на "Unknown"
#feature_names = preprocessor.get_feature_names_out()
#unknown_cut_features = [
#    name for name in feature_names
#    if 'stone.cut.name' in name and 'Unknown' in name
#]
#print(unknown_cut_features)


# Метрики оценки качества

y_pred = model_dt.predict(X_test)
print(classification_report(
    y_test,
    y_pred,
    labels=['Low', 'Medium', 'High']
))


# confusion matrix (Матрица ошибок)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=['Low', 'Medium', 'High']
)
print(cm)

# Визуализация heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low', 'Medium', 'High'],
    yticklabels=['Low', 'Medium', 'High']
)

plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.tight_layout()
plt.show()










# Random Forest (Случайный лес)

model_rf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,        # количество деревьев
        max_depth=None,          # деревья растут до оптимальной глубины
        random_state=42,
        n_jobs=-1                # использование всех ядер
    ))
])

# Обучение модели
model_rf.fit(X_train, y_train)

# Предсказания
y_pred_rf = model_rf.predict(X_test)

# Метрики качества
from sklearn.metrics import classification_report

print("Random Forest classification report:")
print(classification_report(
    y_test,
    y_pred_rf,
    labels=['Low', 'Medium', 'High']
))







# Gradient Boosting (Градиентный бустинг)

model_gb = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', GradientBoostingClassifier(
        n_estimators=200,      # количество деревьев
        learning_rate=0.1,     # скорость обучения
        max_depth=3,           # глубина базовых деревьев
        random_state=42
    ))
])

# Обучение модели
model_gb.fit(X_train, y_train)

# Предсказания
y_pred_gb = model_gb.predict(X_test)

# Метрики качества
print("Gradient Boosting classification report:")
print(classification_report(
    y_test,
    y_pred_gb,
    labels=['Low', 'Medium', 'High']
))
