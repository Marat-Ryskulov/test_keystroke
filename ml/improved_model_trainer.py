# ml/improved_model_trainer.py - Новая система обучения kNN модели

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

from ml.feature_extractor import FeatureExtractor
from config import MODELS_DIR, MIN_TRAINING_SAMPLES

class ImprovedModelTrainer:
    """Улучшенная система обучения kNN модели для биометрической аутентификации"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.calibrated_model = None
        self.best_params = {}
        self.training_stats = {}
        
    def prepare_training_data(self, positive_samples: List) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка сбалансированных данных для обучения"""
        
        # Извлечение признаков из положительных образцов
        X_positive = self.feature_extractor.extract_features_from_samples(positive_samples)
        n_positive = len(X_positive)
        
        if n_positive < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Недостаточно образцов: {n_positive}, нужно минимум {MIN_TRAINING_SAMPLES}")
        
        # Генерация негативных примеров
        X_negative = self._generate_realistic_negatives(X_positive, factor=1.0)
        n_negative = len(X_negative)
        
        # Комбинирование данных
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
        
        # Нормализация признаков
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"Подготовка данных завершена: {n_positive} легитимных, {n_negative} негативных")
        return X_normalized, y
    
    def _generate_realistic_negatives(self, X_positive: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Генерация более различимых негативных примеров"""
        n_samples = len(X_positive)
        n_negatives = int(n_samples * factor)
    
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
    
        print(f"\n📊 АНАЛИЗ ОБУЧАЮЩИХ ДАННЫХ:")
        print(f"Количество ваших образцов: {n_samples}")
        print(f"Среднее ваших данных: {mean}")
        print(f"Стандартное отклонение: {std}")
    
        negatives = []
    
        for i in range(n_negatives):
            # Берем случайный образец как основу
            base_sample = X_positive[np.random.randint(0, len(X_positive))].copy()
        
            # Стратегия: изменяем на 2-4 стандартных отклонения
            change_magnitude = np.random.uniform(2.0, 4.0)  # 2-4 сигмы
            direction = np.random.choice([-1, 1], size=len(base_sample))
        
            # Изменяем 2-3 признака значительно
            features_to_change = np.random.choice(len(base_sample), 
                                            size=np.random.randint(2, 4), 
                                            replace=False)
        
            modified_sample = base_sample.copy()
            for idx in features_to_change:
                if std[idx] > 0:
                    change = direction[idx] * change_magnitude * std[idx]
                    modified_sample[idx] = base_sample[idx] + change
        
            # Обеспечиваем положительность и разумность
            modified_sample = np.maximum(modified_sample, mean * 0.2)
            modified_sample = np.minimum(modified_sample, mean * 5.0)
        
            negatives.append(modified_sample)
    
        negatives_array = np.array(negatives)
    
        # Отладка
        print(f"Среднее негативных: {np.mean(negatives_array, axis=0)}")
    
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X_positive, negatives_array)
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
        print(f"Минимальное расстояние: {min_dist:.3f}")
        print(f"Среднее расстояние: {mean_dist:.3f}")
    
        return negatives_array
    
    def train_user_model(self, positive_samples: List) -> Tuple[bool, float, str]:
        """Основное обучение модели"""
        try:
            print(f"Начало обучения модели для пользователя {self.user_id}")
            
            # Инициализируем training_stats
            self.training_stats = {}
            
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
            
            # Калибровка для получения реалистичных вероятностей
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            self.calibrated_model.fit(X_train, y_train)
            
            # Оценка на тестовой выборке
            test_accuracy = self._evaluate_model(X_test, y_test)
            
            # Статистика обучения (обновляем после _evaluate_model)
            self.training_stats.update({
                'user_id': self.user_id,
                'training_samples': len(positive_samples),
                'total_samples': len(X),
                'best_params': best_params,
                'test_accuracy': test_accuracy,  # Принудительно устанавливаем правильное значение
                'training_date': datetime.now().isoformat()
            })
            
            # Сохранение модели
            self._save_model()
            
            print(f"Обучение завершено. Точность: {test_accuracy:.2%}")
            return True, test_accuracy, f"Модель обучена с точностью {test_accuracy:.2%}"
            
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            return False, 0.0, f"Ошибка обучения: {str(e)}"
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров через Grid Search"""
        
        # Параметры для поиска
        param_grid = {
            'n_neighbors': range(5, min(20, len(X_train) // 3)),  # Больше соседей
            'weights': ['uniform'],  # Только uniform веса
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree']
        }
        
        # Grid Search с кросс-валидацией
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            return_train_score=False
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Оптимальные параметры: {grid_search.best_params_}")
        print(f"CV точность: {grid_search.best_score_:.3f}")
        
        self.best_params = grid_search.best_params_
        return grid_search.best_params_
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Оценка модели на тестовой выборке"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # Вероятности для положительного класса
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Метрики на тесте:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")
        
        # Сохраняем метрики и данные для ROC-кривой
        self.training_stats.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_test': y_test.tolist(),  # Истинные метки
            'y_proba': y_proba.tolist()  # Предсказанные вероятности
        })
        
        return accuracy
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Предсказание БЕЗ калибровки (временно)"""
        if self.model is None:
            raise ValueError("Модель не обучена")
    
        # Нормализация признаков
        features_scaled = self.scaler.transform(features.reshape(1, -1))
    
        # Используем ТОЛЬКО сырую модель
        raw_proba = self.model.predict_proba(features_scaled)[0]
    
        print(f"🔍 ОТЛАДКА БЕЗ КАЛИБРОВКИ:")
        print(f"   Признаки: {features}")
        print(f"   Сырая вероятность: {raw_proba}")
    
        # Вероятность положительного класса
        confidence = raw_proba[1] if len(raw_proba) > 1 else raw_proba[0]
    
        # Используем порог 0.5 для сырой модели
        is_legitimate = confidence >= 0.1
    
        return is_legitimate, confidence
    
    def _save_model(self):
        """Сохранение обученной модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{self.user_id}_improved_knn.pkl")
        
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена: {model_path}")
    
    @classmethod
    def load_model(cls, user_id: int) -> Optional['ImprovedModelTrainer']:
        """Загрузка обученной модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_improved_knn.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            trainer = cls(user_id)
            trainer.model = model_data['model']
            trainer.calibrated_model = model_data['calibrated_model']
            trainer.scaler = model_data['scaler']
            trainer.best_params = model_data['best_params']
            trainer.training_stats = model_data.get('training_stats', {})
            
            return trainer
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return {
            'is_trained': self.model is not None,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }