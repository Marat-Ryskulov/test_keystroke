# ml/simple_knn_trainer.py - Простая и надежная система обучения

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

from ml.feature_extractor import FeatureExtractor
from config import MODELS_DIR, MIN_TRAINING_SAMPLES

class SimpleKNNTrainer:
    """Простая и надежная система обучения kNN модели"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = {}
        self.training_stats = {}
        
    def prepare_training_data(self, positive_samples: List) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных с качественными негативными примерами"""
        
        # Извлечение признаков из положительных образцов
        X_positive = self.feature_extractor.extract_features_from_samples(positive_samples)
        n_positive = len(X_positive)
        
        if n_positive < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Недостаточно образцов: {n_positive}, нужно минимум {MIN_TRAINING_SAMPLES}")
        
        # Генерация негативных примеров
        X_negative = self._generate_quality_negatives(X_positive)
        n_negative = len(X_negative)
        
        # Комбинирование данных
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
        
        # Нормализация признаков
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"Данные подготовлены: {n_positive} ваших, {n_negative} негативных")
        return X_normalized, y
    
    def _generate_quality_negatives(self, X_positive: np.ndarray) -> np.ndarray:
        """Генерация качественных негативных примеров"""
        n_samples = len(X_positive)
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        
        # Обеспечиваем минимальную вариативность
        std = np.maximum(std, mean * 0.1)
        
        negatives = []
        
        # Стратегия 1: Медленная печать (30%)
        slow_count = int(n_samples * 0.3)
        for _ in range(slow_count):
            sample = mean.copy()
            # Увеличиваем времена, уменьшаем скорость
            sample[0] *= np.random.uniform(1.5, 2.5)  # avg_dwell_time
            sample[2] *= np.random.uniform(1.8, 3.0)  # avg_flight_time  
            sample[4] *= np.random.uniform(0.4, 0.7)  # typing_speed
            sample[5] *= np.random.uniform(1.5, 2.5)  # total_typing_time
            
            # Добавляем шум
            noise = np.random.normal(0, std * 0.3)
            sample += noise
            sample = np.maximum(sample, mean * 0.1)
            negatives.append(sample)
        
        # Стратегия 2: Быстрая печать (30%)
        fast_count = int(n_samples * 0.3)
        for _ in range(fast_count):
            sample = mean.copy()
            # Уменьшаем времена, увеличиваем скорость
            sample[0] *= np.random.uniform(0.3, 0.6)  # avg_dwell_time
            sample[2] *= np.random.uniform(0.2, 0.5)  # avg_flight_time
            sample[4] *= np.random.uniform(1.8, 3.5)  # typing_speed
            sample[5] *= np.random.uniform(0.3, 0.7)  # total_typing_time
            
            # Добавляем шум
            noise = np.random.normal(0, std * 0.3)
            sample += noise
            sample = np.maximum(sample, mean * 0.1)
            negatives.append(sample)
        
        # Стратегия 3: Нестабильная печать (40%)
        unstable_count = n_samples - slow_count - fast_count
        for _ in range(unstable_count):
            sample = mean.copy()
            # Сильно увеличиваем вариативность
            sample[1] *= np.random.uniform(3.0, 8.0)  # std_dwell_time
            sample[3] *= np.random.uniform(3.0, 8.0)  # std_flight_time
            
            # Случайно изменяем средние значения
            sample[0] *= np.random.uniform(0.5, 2.0)
            sample[2] *= np.random.uniform(0.5, 2.0)
            sample[4] *= np.random.uniform(0.6, 1.8)
            sample[5] *= np.random.uniform(0.8, 1.5)
            
            # Добавляем шум
            noise = np.random.normal(0, std * 0.5)
            sample += noise
            sample = np.maximum(sample, mean * 0.05)
            negatives.append(sample)
        
        negatives_array = np.array(negatives)
        
        # Проверяем качество разделения
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X_positive, negatives_array)
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
        
        print(f"Качество разделения: мин={min_dist:.2f}, среднее={mean_dist:.2f}")
        
        return negatives_array
    
    def train_user_model(self, positive_samples: List) -> Tuple[bool, float, str]:
        """Основное обучение модели"""
        try:
            print(f"Начало обучения для пользователя {self.user_id}")
            
            # Подготовка данных
            X, y = self.prepare_training_data(positive_samples)
            
            # Разделение на обучение и тест
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            # Поиск оптимальных параметров
            best_params = self._optimize_hyperparameters(X_train, y_train)
            
            # Обучение финальной модели
            self.model = KNeighborsClassifier(**best_params)
            self.model.fit(X_train, y_train)
            
            # Оценка на тестовой выборке
            test_accuracy = self._evaluate_model(X_test, y_test)
            
            # Сохранение статистики
            self.training_stats = {
                'user_id': self.user_id,
                'training_samples': len(positive_samples),
                'total_samples': len(X),
                'best_params': best_params,
                'test_accuracy': test_accuracy,
                'training_date': datetime.now().isoformat()
            }
            
            # Сохранение модели
            self._save_model()
            
            print(f"Обучение завершено. Точность: {test_accuracy:.2%}")
            return True, test_accuracy, f"Модель обучена с точностью {test_accuracy:.2%}"
            
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            return False, 0.0, f"Ошибка обучения: {str(e)}"
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров"""
        
        # Ограниченный набор параметров для стабильности
        param_grid = {
            'n_neighbors': range(3, min(12, len(X_train) // 6)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Grid Search с 5-fold кросс-валидацией
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"CV точность: {grid_search.best_score_:.3f}")
        
        self.best_params = grid_search.best_params_
        return grid_search.best_params_
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Оценка модели"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Метрики:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")
        
        # Сохраняем метрики и данные для ROC
        self.training_stats.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_test': y_test.tolist(),
            'y_proba': y_proba.tolist()
        })
        
        return accuracy
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Простое предсказание с отладкой"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Нормализация признаков
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Получение расстояний и соседей
        distances, indices = self.model.kneighbors(features_scaled)
        
        # Получение вероятности
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = proba[1] if len(proba) > 1 else proba[0]
        
        print(f"\n🔍 ПОЛНАЯ ОТЛАДКА:")
        print(f"   Исходные признаки: {features}")
        print(f"   Нормализованные: {features_scaled[0]}")
        print(f"   Модель k: {self.model.n_neighbors}")
        print(f"   Веса: {self.model.weights}")
        print(f"   Расстояния до соседей: {distances[0]}")
        print(f"   Индексы соседей: {indices[0]}")
        print(f"   Сырая вероятность: {proba}")
        print(f"   Уверенность: {confidence:.3f}")
        
        # Принятие решения с порогом 75%
        is_legitimate = confidence >= 0.75
        
        print(f"   Порог: 0.75")
        print(f"   Результат: {'ПРИНЯТ' if is_legitimate else 'ОТКЛОНЕН'}")
        
        return is_legitimate, confidence
    
    def _save_model(self):
        """Сохранение модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{self.user_id}_simple_knn.pkl")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена: {model_path}")
    
    @classmethod
    def load_model(cls, user_id: int) -> Optional['SimpleKNNTrainer']:
        """Загрузка модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_simple_knn.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            trainer = cls(user_id)
            trainer.model = model_data['model']
            trainer.scaler = model_data['scaler']
            trainer.best_params = model_data['best_params']
            trainer.training_stats = model_data.get('training_stats', {})
            
            return trainer
            
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            'is_trained': self.model is not None,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }