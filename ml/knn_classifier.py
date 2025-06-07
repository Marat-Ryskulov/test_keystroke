# ml/knn_classifier.py - Исправленная версия с более реалистичной уверенностью

import numpy as np
from typing import Tuple, Optional, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

from config import KNN_NEIGHBORS, MODELS_DIR
import os

class KNNAuthenticator:
    """KNN классификатор для аутентификации по динамике нажатий"""
    

    def __init__(self, n_neighbors: int = KNN_NEIGHBORS):
        self.n_neighbors = min(n_neighbors, 5)  # Увеличим до 5 для стабильности
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric='euclidean',
            weights='distance',     # Ближайшие соседи важнее
            algorithm='ball_tree'   # Более точный алгоритм
        )
        self.is_trained = False
        self.normalization_stats = None
        self.training_data = None
        
    def train(self, X_positive: np.ndarray, X_negative: np.ndarray = None) -> Tuple[bool, float]:
        """Обучение с более сбалансированным подходом"""
        n_samples = len(X_positive)
        if n_samples < 5:
            return False, 0.0
    
        print(f"\n🎯 СБАЛАНСИРОВАННОЕ ОБУЧЕНИЕ")
        print(f"Твоих образцов: {n_samples}")
    
        self.training_data = X_positive.copy()
    
        # Генерируем негативные примеры
        if X_negative is None or len(X_negative) == 0:
            X_negative = self._generate_balanced_negatives(X_positive)
    
        # ИСПРАВЛЕНИЕ: Более сбалансированное соотношение
        # Соотношение примерно 2:1 в пользу твоих данных (было 3:1 или 4:1)
        neg_count = max(n_samples // 2, 10)  # Минимум 10, обычно в 2 раза меньше
    
        if len(X_negative) > neg_count:
            # Берем разнообразные негативные примеры (не только самые далекие)
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X_negative, X_positive)
            min_distances = np.min(distances, axis=1)
            
            # Берем 50% далеких + 50% средних (не только экстремальные)
            far_count = neg_count // 2
            medium_count = neg_count - far_count
            
            # Далекие примеры
            far_indices = np.argsort(min_distances)[-far_count:]
            
            # Средние примеры (исключая уже выбранные далекие)
            remaining_indices = np.setdiff1d(np.arange(len(X_negative)), far_indices)
            remaining_distances = min_distances[remaining_indices]
            medium_indices = remaining_indices[np.argsort(remaining_distances)[len(remaining_distances)//4:len(remaining_distances)//4+medium_count]]
            
            selected_indices = np.concatenate([far_indices, medium_indices])
            X_negative = X_negative[selected_indices]
    
        print(f"Финальные данные: {len(X_positive)} ТВОИХ vs {len(X_negative)} ЧУЖИХ")
        print(f"Соотношение: {len(X_positive)/(len(X_positive)+len(X_negative))*100:.0f}% твоих данных")
    
        # Обучение
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(len(X_positive)), np.zeros(len(X_negative))])
    
        self.model.fit(X, y)
        self.is_trained = True
    
        # Проверяем качество
        train_accuracy = self.model.score(X, y)
        print(f"Train accuracy: {train_accuracy:.3f}")

        self.test_data = {
            'X_positive': X_positive.copy(),
            'X_negative': X_negative.copy() if X_negative is not None else None,
            'y_positive': np.ones(len(X_positive)),
            'y_negative': np.zeros(len(X_negative)) if X_negative is not None else None
        }
    
        return True, train_accuracy
    
    def authenticate(self, features: np.ndarray, threshold: float = 0.5, verbose: bool = False) -> Tuple[bool, float, dict]:
        """
        ИСПРАВЛЕННАЯ аутентификация с более реалистичной уверенностью
        """
        if not self.is_trained:
            return False, 0.0, {}

        # Убеждаемся, что features - это 1D массив
        if features.ndim > 1:
            features = features.flatten()

        if verbose:
            print(f"\n=== НАЧАЛО АУТЕНТИФИКАЦИИ ===")
            print(f"Входящие признаки: {features}")

        # 1. Основное предсказание KNN с ПЛАВНОЙ уверенностью
        features_reshaped = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features_reshaped)[0]

        # Получаем базовую вероятность
        knn_probability = 0.5  # По умолчанию 50%
        if len(probabilities) > 1 and 1.0 in self.model.classes_:
            class_1_index = list(self.model.classes_).index(1.0)
            raw_prob = probabilities[class_1_index]
            
            # ИСПРАВЛЕНИЕ: Применяем сглаживание чтобы избежать экстремальных значений
            # Ограничиваем вероятность в диапазоне 10%-90%
            knn_probability = np.clip(raw_prob, 0.1, 0.9)
            
            # Дополнительное сглаживание к центру
            knn_probability = 0.5 + (knn_probability - 0.5) * 0.8

        # 2. Анализ расстояний с более плавной функцией
        distance_score = 0.5  # По умолчанию
        distance_details = {}

        if hasattr(self, 'training_data') and self.training_data is not None:
            from sklearn.metrics.pairwise import euclidean_distances
    
            X_positive = self.training_data
            distances = euclidean_distances(features_reshaped, X_positive)[0]
    
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
    
            # Статистика обучающих данных
            if len(X_positive) > 1:
                train_distances = euclidean_distances(X_positive, X_positive)
                train_distances = train_distances[train_distances > 0]
                mean_train_distance = np.mean(train_distances)
                std_train_distance = np.std(train_distances)
            else:
                mean_train_distance = 1.0
                std_train_distance = 0.5
    
            # ИСПРАВЛЕНИЕ: Более плавная функция расстояния
            norm_min = min_distance / (mean_train_distance + 1e-6)
            
            # Используем sigmoid-подобную функцию вместо резкого обрезания
            distance_score = 1.0 / (1.0 + np.exp(norm_min - 1.5))  # sigmoid с центром в 1.5
            distance_score = np.clip(distance_score, 0.1, 0.9)  # Ограничиваем диапазон
    
            distance_details = {
                'min_distance': min_distance,
                'mean_distance': mean_distance,
                'mean_train_distance': mean_train_distance,
                'normalized_distance': norm_min
            }

        # 3. Анализ признаков с более мягкими штрафами
        feature_score = 0.7  # По умолчанию хорошая оценка
        feature_details = {}

        if hasattr(self, 'training_data') and self.training_data is not None:
            X_positive = self.training_data
            train_mean = np.mean(X_positive, axis=0)
            train_std = np.std(X_positive, axis=0)
    
            total_penalty = 0
            feature_penalties = []
            feature_names = ['avg_dwell', 'std_dwell', 'avg_flight', 'std_flight', 'speed', 'total_time']
    
            for i, (feat_val, train_m, train_s, name) in enumerate(zip(features, train_mean, train_std, feature_names)):
                if train_s > 0:
                    z_score_val = abs(feat_val - train_m) / train_s
                    if hasattr(z_score_val, '__len__'):
                        z_score_val = float(z_score_val)
                    
                    # ИСПРАВЛЕНИЕ: Более мягкие штрафы
                    # Используем sigmoid для плавного перехода
                    if z_score_val <= 1.0:
                        penalty = 0  # В пределах 1 стандартного отклонения - без штрафа
                    elif z_score_val <= 2.0:
                        penalty = 0.05 * (z_score_val - 1.0)  # Небольшой штраф 1-2 sigma
                    elif z_score_val <= 3.0:
                        penalty = 0.05 + 0.1 * (z_score_val - 2.0)  # Средний штраф 2-3 sigma
                    else:
                        penalty = 0.15 + 0.15 * min(z_score_val - 3.0, 2.0)  # Максимум 30% штрафа
                    
                    penalty = min(penalty, 0.3)  # Ограничиваем максимальный штраф
                else:
                    penalty = 0.0
                    z_score_val = 0.0
        
                feature_penalties.append(penalty)
                feature_details[name] = {
                    'value': float(feat_val),
                    'train_mean': float(train_m),
                    'train_std': float(train_s),
                    'z_score': float(z_score_val),
                    'penalty': penalty
                }
    
            # Применяем штрафы более мягко
            total_penalty = np.mean(feature_penalties)  # Среднее вместо суммы
            feature_score = max(0.2, 1.0 - total_penalty)  # Минимум 20% вместо 10%

        # 4. ИСПРАВЛЕННОЕ комбинирование оценок
        # Более консервативные веса для избежания экстремальных результатов
        if len(self.training_data) >= 30:
            weights = {'knn': 0.4, 'distance': 0.35, 'features': 0.25}
        elif len(self.training_data) >= 15:
            weights = {'knn': 0.35, 'distance': 0.4, 'features': 0.25}
        else:
            weights = {'knn': 0.3, 'distance': 0.5, 'features': 0.2}

        final_probability = (
            weights['knn'] * knn_probability +
            weights['distance'] * distance_score +
            weights['features'] * feature_score
        )
        
        # ИСПРАВЛЕНИЕ: Дополнительное сглаживание финального результата
        # Избегаем крайних значений 0% и 100%
        final_probability = np.clip(final_probability, 0.05, 0.95)
        
        # Принятие решения с более разумным порогом
        is_authenticated = final_probability >= threshold

        # Детальная статистика
        detailed_stats = {
            'knn_confidence': knn_probability,
            'distance_score': distance_score,
            'feature_score': feature_score,
            'final_confidence': final_probability,
            'threshold': threshold,
            'weights': weights,
            'distance_details': distance_details,
            'feature_details': feature_details,
            'training_samples': len(self.training_data) if hasattr(self, 'training_data') else 0
        }

        if verbose:
            print(f"\nФИНАЛЬНЫЕ ОЦЕНКИ:")
            print(f"  KNN: {knn_probability:.3f} (вес: {weights['knn']})")
            print(f"  Distance: {distance_score:.3f} (вес: {weights['distance']})")
            print(f"  Features: {feature_score:.3f} (вес: {weights['features']})")
            print(f"  Final: {final_probability:.3f} (порог: {threshold})")
            print(f"  РЕЗУЛЬТАТ: {'ПРИНЯТ' if is_authenticated else 'ОТКЛОНЕН'}")
            print(f"=== КОНЕЦ АУТЕНТИФИКАЦИИ ===\n")

        return is_authenticated, final_probability, detailed_stats
    
    def _generate_balanced_negatives(self, X_positive: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Генерация СБАЛАНСИРОВАННЫХ негативных примеров"""
        n_samples = len(X_positive)
        n_features = X_positive.shape[1]

        # Анализируем ТВОИ данные
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        
        # Обеспечиваем минимальную вариативность
        std = np.maximum(std, mean * 0.1)

        print(f"\n🔍 СОЗДАНИЕ СБАЛАНСИРОВАННЫХ НЕГАТИВОВ:")
        print(f"  Удержание клавиш: {mean[0]*1000:.1f} ± {std[0]*1000:.1f} мс")
        print(f"  Время между клавишами: {mean[2]*1000:.1f} ± {std[2]*1000:.1f} мс")
        print(f"  Скорость печати: {mean[4]:.1f} ± {std[4]:.1f} кл/с")

        synthetic_samples = []

        # Стратегия 1: Близкие варианты (40%) - НЕ слишком далеко
        close_count = int(n_samples * 0.4)
        print(f"Создаем {close_count} БЛИЗКИХ вариантов...")
        for i in range(close_count):
            sample = mean.copy()
            
            # Изменяем 1-2 признака умеренно
            features_to_change = np.random.choice(6, size=np.random.randint(1, 3), replace=False)
            
            for feat_idx in features_to_change:
                # Умеренные изменения: 70%-130% от среднего
                factor = np.random.uniform(0.7, 1.3)
                sample[feat_idx] = mean[feat_idx] * factor
            
            # Небольшой шум
            noise = np.random.normal(0, std * 0.4)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.1)
            synthetic_samples.append(sample)

        # Стратегия 2: Другой стиль (40%)
        different_style_count = int(n_samples * 0.4)
        print(f"Создаем {different_style_count} с ДРУГИМ стилем...")
        for i in range(different_style_count):
            if np.random.random() < 0.5:
                # Быстрые печатающие (но не экстремально)
                style_factors = np.array([
                    np.random.uniform(0.6, 0.9),     # быстрее удержание
                    np.random.uniform(0.7, 1.2),     # вариативность
                    np.random.uniform(0.5, 0.8),     # быстрее переходы
                    np.random.uniform(0.6, 1.3),     # вариативность пауз
                    np.random.uniform(1.1, 1.8),     # выше скорость
                    np.random.uniform(0.6, 0.9)      # меньше времени
                ])
            else:
                # Медленные печатающие (но не экстремально)
                style_factors = np.array([
                    np.random.uniform(1.1, 1.6),     # медленнее удержание
                    np.random.uniform(0.8, 1.5),     # вариативность
                    np.random.uniform(1.2, 2.0),     # медленнее переходы
                    np.random.uniform(1.0, 1.8),     # вариативность пауз
                    np.random.uniform(0.5, 0.9),     # ниже скорость
                    np.random.uniform(1.1, 1.7)      # больше времени
                ])
            
            sample = mean * style_factors
            noise = np.random.normal(0, std * 0.3)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.05)
            synthetic_samples.append(sample)

        # Стратегия 3: Умеренно далекие (20%)
        far_count = n_samples - close_count - different_style_count
        print(f"Создаем {far_count} УМЕРЕННО далеких...")
        for i in range(far_count):
            # Более заметные, но не экстремальные отличия
            factors = np.random.uniform(0.4, 2.5, size=6)
            sample = mean * factors
            noise = np.random.normal(0, std * 0.6)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.02)
            synthetic_samples.append(sample)

        result = np.array(synthetic_samples)
        print(f"  Создано {len(result)} сбалансированных негативных образцов")
        return result
    
    def save_model(self, user_id: int):
        """Сохранение модели на диск"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
    
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
    
        model_data = {
            'model': self.model,
            'n_neighbors': self.n_neighbors,
            'normalization_stats': self.normalization_stats,
            'is_trained': self.is_trained,
            'training_data': self.training_data,
            'test_data': self.test_data
        }
    
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, user_id: int) -> Optional['KNNAuthenticator']:
        """Загрузка модели с диска"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
    
        if not os.path.exists(model_path):
            return None
    
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
            authenticator = cls(n_neighbors=model_data['n_neighbors'])
            authenticator.model = model_data['model']
            authenticator.normalization_stats = model_data['normalization_stats']
            authenticator.is_trained = model_data['is_trained']
            authenticator.training_data = model_data.get('training_data', None)
            authenticator.test_data = model_data.get('test_data', None)
        
            return authenticator
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Получение важности признаков (для анализа)"""
        if not self.is_trained:
            return []
        
        feature_names = [
            'avg_dwell_time',
            'std_dwell_time', 
            'avg_flight_time',
            'std_flight_time',
            'typing_speed',
            'total_typing_time'
        ]
        
        # Получаем обучающие данные
        X_train = self.model._fit_X
        
        # Вычисляем дисперсию для каждого признака
        variances = np.var(X_train, axis=0)
        
        # Нормализуем важности
        importance = variances / np.sum(variances)
        
        return list(zip(feature_names, importance))