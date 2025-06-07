# gui/__init__.py - Обновленный список импортов
from .main_window import MainWindow
from .login_window import LoginWindow
from .register_window import RegisterWindow
from .training_window import TrainingWindow
from .diploma_stats_window import DiplomaStatsWindow  # Новое упрощенное окно

__all__ = [
    'MainWindow', 
    'LoginWindow', 
    'RegisterWindow', 
    'TrainingWindow', 
    'DiplomaStatsWindow'
]