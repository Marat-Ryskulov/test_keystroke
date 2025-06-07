# gui/simplified_stats_window.py - Статистика только с распределениями

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List
from datetime import datetime
import json

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from config import FONT_FAMILY

plt.style.use('default')

class SimplifiedStatsWindow:
    """Упрощенная статистика - только распределения признаков"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.model_manager = ModelManager()
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Статистика клавиатурного почерка - {user.username}")
        self.window.geometry("1000x700")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Получение данных
        self.training_samples = self.db.get_user_training_samples(user.id)
        
        # Создание интерфейса
        self.create_interface()
        self.load_statistics()
    
    def create_interface(self):
        """Создание интерфейса"""
        # Заголовок
        header_frame = ttk.Frame(self.window, padding=10)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text=f"Статистика клавиатурного почерка - {self.user.username}",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack()
        
        # Основная информация
        info_frame = ttk.LabelFrame(header_frame, text="Информация", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(info_frame, height=4, width=100, font=(FONT_FAMILY, 10))
        self.info_text.pack()
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Вкладка: Распределения признаков
        self.create_features_tab()
        
        # Вкладка: Временной анализ
        self.create_temporal_tab()
        
        # Кнопки
        self.create_buttons()
    
    def create_features_tab(self):
        """Вкладка распределений признаков"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Распределения признаков")
        
        # График признаков (2x2)
        self.fig_features, ((self.ax_f1, self.ax_f2), (self.ax_f3, self.ax_f4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas_features = FigureCanvasTkAgg(self.fig_features, frame)
        self.canvas_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_temporal_tab(self):
        """Вкладка временного анализа"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Временной анализ")
        
        # Графики времени (1x2)
        self.fig_time, (self.ax_t1, self.ax_t2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, frame)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_buttons(self):
        """Создание кнопок"""
        buttons_frame = ttk.Frame(self.window)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="Экспорт данных",
            command=self.export_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Обновить",
            command=self.refresh_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Закрыть",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def load_statistics(self):
        """Загрузка статистики"""
        try:
            self.load_general_info()
            self.load_features_analysis()
            self.load_temporal_analysis()
            
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
            import traceback
            traceback.print_exc()
    
    def load_general_info(self):
        """Загрузка общей информации"""
        n_samples = len(self.training_samples)
        
        if n_samples == 0:
            info = "Нет данных для анализа"
            self.info_text.insert(tk.END, info)
            return
        
        # Извлечение признаков для анализа
        features_data = []
        for sample in self.training_samples:
            if sample.features:
                features_data.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('typing_speed', 0),
                    sample.features.get('total_typing_time', 0)
                ])
        
        if not features_data:
            info = "Признаки не рассчитаны для образцов"
            self.info_text.insert(tk.END, info)
            return
        
        features_array = np.array(features_data)
        
        # Статистика
        info = f"""Пользователь: {self.user.username}
Дата регистрации: {self.user.created_at.strftime('%d.%m.%Y %H:%M') if self.user.created_at else 'Не указана'}
Статус модели: {'Обучена' if self.user.is_trained else 'Не обучена'}
Количество образцов: {n_samples}

Характеристики клавиатурного почерка:
Время удержания клавиш: {np.mean(features_array[:, 0])*1000:.1f} ± {np.std(features_array[:, 0])*1000:.1f} мс
Время между клавишами: {np.mean(features_array[:, 1])*1000:.1f} ± {np.std(features_array[:, 1])*1000:.1f} мс  
Скорость печати: {np.mean(features_array[:, 2]):.1f} ± {np.std(features_array[:, 2]):.1f} клавиш/сек
Общее время ввода: {np.mean(features_array[:, 3]):.1f} ± {np.std(features_array[:, 3]):.1f} сек"""
        
        self.info_text.insert(tk.END, info)
    
    def load_features_analysis(self):
        """Анализ распределений признаков"""
        if not self.training_samples:
            for ax in [self.ax_f1, self.ax_f2, self.ax_f3, self.ax_f4]:
                ax.text(0.5, 0.5, 'Нет данных для анализа', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            self.canvas_features.draw()
            return
        
        try:
            # Извлечение данных признаков
            features_data = []
            for sample in self.training_samples:
                if sample.features:
                    features_data.append([
                        sample.features.get('avg_dwell_time', 0) * 1000,  # в мс
                        sample.features.get('avg_flight_time', 0) * 1000,  # в мс
                        sample.features.get('typing_speed', 0),
                        sample.features.get('total_typing_time', 0)
                    ])
            
            if not features_data:
                return
            
            features_array = np.array(features_data)
            feature_names = [
                'Время удержания клавиш (мс)', 
                'Время между клавишами (мс)', 
                'Скорость печати (клавиш/сек)', 
                'Общее время ввода (сек)'
            ]
            
            # Четыре графика распределений
            axes = [self.ax_f1, self.ax_f2, self.ax_f3, self.ax_f4]
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
            
            for i, (ax, name, color) in enumerate(zip(axes, feature_names, colors)):
                data = features_array[:, i]
                
                # Гистограмма
                ax.hist(data, bins=min(15, len(data)//2 + 1), alpha=0.7, 
                       color=color, edgecolor='black')
                
                # Статистики
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Среднее: {mean_val:.2f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', 
                          alpha=0.7, label=f'±σ: {std_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
                
                ax.set_xlabel(name, fontsize=10)
                ax.set_ylabel('Частота', fontsize=10)
                ax.set_title(f'Распределение: {name}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            self.fig_features.tight_layout()
            self.canvas_features.draw()
            
        except Exception as e:
            print(f"Ошибка анализа признаков: {e}")
    
    def load_temporal_analysis(self):
        """Временной анализ активности"""
        if not self.training_samples:
            for ax in [self.ax_t1, self.ax_t2]:
                ax.text(0.5, 0.5, 'Нет данных для анализа', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            self.canvas_time.draw()
            return
        
        try:
            # График 1: Распределение по времени суток
            timestamps = [sample.timestamp for sample in self.training_samples]
            hours = [t.hour for t in timestamps]
            
            self.ax_t1.hist(hours, bins=24, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax_t1.set_xlabel('Час дня')
            self.ax_t1.set_ylabel('Количество образцов')
            self.ax_t1.set_title('Активность по времени суток')
            self.ax_t1.grid(True, alpha=0.3)
            
            # График 2: Сбор данных по дням
            dates = [t.date() for t in timestamps]
            unique_dates = sorted(set(dates))
            
            if len(unique_dates) > 1:
                daily_counts = [dates.count(date) for date in unique_dates]
                self.ax_t2.plot(unique_dates, daily_counts, 'o-', color='green', linewidth=2, markersize=6)
                self.ax_t2.set_xlabel('Дата')
                self.ax_t2.set_ylabel('Образцов в день')
                self.ax_t2.set_title('Сбор данных по дням')
                self.ax_t2.tick_params(axis='x', rotation=45)
                self.ax_t2.grid(True, alpha=0.3)
            else:
                self.ax_t2.text(0.5, 0.5, 'Все образцы собраны в один день', 
                               ha='center', va='center', transform=self.ax_t2.transAxes, fontsize=12)
            
            self.fig_time.tight_layout()
            self.canvas_time.draw()
            
        except Exception as e:
            print(f"Ошибка временного анализа: {e}")
    
    def refresh_data(self):
        """Обновление данных"""
        try:
            # Перезагружаем данные
            self.training_samples = self.db.get_user_training_samples(self.user.id)
            
            # Очищаем и перезагружаем
            self.info_text.delete('1.0', tk.END)
            
            # Очищаем графики
            for ax in [self.ax_f1, self.ax_f2, self.ax_f3, self.ax_f4]:
                ax.clear()
            for ax in [self.ax_t1, self.ax_t2]:
                ax.clear()
            
            # Перезагружаем статистику
            self.load_statistics()
            
            messagebox.showinfo("Обновление", "Данные обновлены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обновления: {e}")
    
    def export_data(self):
        """Экспорт данных в файл"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
                title="Экспорт данных"
            )
            
            if filename:
                if filename.endswith('.csv'):
                    self.export_to_csv(filename)
                else:
                    self.export_to_json(filename)
                
                messagebox.showinfo("Экспорт", f"Данные экспортированы: {filename}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def export_to_json(self, filename: str):
        """Экспорт в JSON"""
        data = {
            'user': self.user.username,
            'export_date': datetime.now().isoformat(),
            'total_samples': len(self.training_samples),
            'samples': []
        }
        
        for i, sample in enumerate(self.training_samples):
            sample_data = {
                'sample_id': i + 1,
                'timestamp': sample.timestamp.isoformat(),
                'features': sample.features
            }
            data['samples'].append(sample_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_to_csv(self, filename: str):
        """Экспорт в CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'sample_id', 'timestamp', 
                'avg_dwell_time', 'std_dwell_time',
                'avg_flight_time', 'std_flight_time',
                'typing_speed', 'total_typing_time'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, sample in enumerate(self.training_samples):
                row = {
                    'sample_id': i + 1,
                    'timestamp': sample.timestamp.isoformat(),
                    **sample.features
                }
                writer.writerow(row)