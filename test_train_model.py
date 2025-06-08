import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix

# Предположим, 'your_typing_data' - это твой набор признаков (50 образцов)
# Например, np.array([[время_нажатия_1, время_между_1_и_2, ...], [...]])
# Каждый внутренний список - это признаки одного образца панграммы.
your_typing_data = np.random.rand(50, 10) # Замени на свои реальные 50 образцов

# Метки для твоего почерка - все 1, так как это "легитимный" пользователь
y_labels_own = np.ones(len(your_typing_data))

# Шаг 1: Разделяем твои данные на обучающую и тестовую выборки
# 40 для обучения/валидации, 10 для финального тестирования на "легитимном"
X_train_own, X_test_own, y_train_own, y_test_own = train_test_split(
    your_typing_data, y_labels_own, test_size=10, random_state=42
)

# Шаг 2: Определяем диапазон k для поиска
param_grid = {'n_neighbors': np.arange(1, 11)} # Пробуем k от 1 до 10

# Шаг 3: Настраиваем kNN классификатор и GridSearchCV
knn = KNeighborsClassifier()

# Используем GridSearchCV для поиска лучшего k с 5-кратной кросс-валидацией
# Обучение будет происходить только на X_train_own
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_own, y_train_own)

# Получаем лучшую модель и лучшее k
best_k = grid_search.best_params_['n_neighbors']
best_knn_model = grid_search.best_estimator_

print(f"Лучшее значение k (n_neighbors) по кросс-валидации: {best_k}")
print(f"Средняя точность на кросс-валидации с лучшим k: {grid_search.best_score_:.4f}")

# Теперь у тебя есть обученная модель best_knn_model, готовая к тестированию.