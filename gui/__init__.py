# gui/__init__.py - Обновленный список импортов
from .main_window import MainWindow
from .login_window import LoginWindow
from .register_window import RegisterWindow
from .training_window import TrainingWindow
from .simplified_stats_window import SimplifiedStatsWindow
from .controlled_testing_window import ControlledTestingWindow

__all__ = [
    'MainWindow', 
    'LoginWindow', 
    'RegisterWindow', 
    'TrainingWindow', 
    'SimplifiedStatsWindow',
    'ControlledTestingWindow'
]