#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система двухфакторной аутентификации с динамикой нажатий клавиш
Главный файл запуска приложения
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь Python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Импортируем главное окно
from gui.main_window import MainWindow

def main():
    """Точка входа в приложение"""
    try:
        print("🚀 Запуск системы двухфакторной аутентификации...")
        print("📁 Инициализация базы данных...")
        
        # Проверяем доступность всех зависимостей
        try:
            import sklearn
            import numpy
            import matplotlib
            print("✅ Все зависимости найдены")
        except ImportError as e:
            print(f"❌ Отсутствует зависимость: {e}")
            print("Установите зависимости: pip install scikit-learn numpy matplotlib")
            input("Нажмите Enter для продолжения...")
            
        # Создаем и запускаем главное окно
        app = MainWindow()
        print("🖥️ Запуск графического интерфейса...")
        app.run()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что установлены все зависимости:")
        print("pip install scikit-learn numpy matplotlib")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("Нажмите Enter для выхода...")
        sys.exit(1)

if __name__ == "__main__":
    main()