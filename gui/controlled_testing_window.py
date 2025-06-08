# gui/controlled_testing_window.py - Исправленное контролируемое тестирование с отдельным окном результатов

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json
import os

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from config import PANGRAM, FONT_FAMILY, FONT_SIZE, DATA_DIR

class ControlledTestingWindow:
    """Окно контролируемого тестирования эффективности системы"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        
        # Данные тестирования
        self.legitimate_samples = []  # Легитимные образцы (обычная скорость)
        self.impostor_samples = []    # Имитационные образцы (измененная скорость)
        
        # Состояние тестирования
        self.current_phase = "legitimate"  # "legitimate", "impostor_slow", "impostor_fast", "completed"
        self.samples_collected = 0
        self.target_samples = 10
        
        # Текущая сессия записи
        self.session_id = None
        self.is_recording = False
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Контролируемое тестирование эффективности")
        self.window.geometry("800x700")  # Уменьшил высоту, убрав место под результаты
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Нормализованный текст
        self.normalized_target = self._normalize_text(PANGRAM)
        
        self.create_interface()
        self.start_testing()
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        return text.lower().replace(" ", "")
    
    def create_interface(self):
        """Создание интерфейса"""
        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Контролируемое тестирование эффективности",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Прогресс тестирования
        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс тестирования", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.phase_label = ttk.Label(
            progress_frame,
            text="",
            font=(FONT_FAMILY, 12, 'bold')
        )
        self.phase_label.pack()
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=500,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
        
        # Инструкции
        instructions_frame = ttk.LabelFrame(main_frame, text="Инструкции", padding=10)
        instructions_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.instructions_label = ttk.Label(
            instructions_frame,
            text="",
            wraplength=600,
            justify=tk.LEFT,
            font=(FONT_FAMILY, 10)
        )
        self.instructions_label.pack()
        
        # Поле ввода
        input_frame = ttk.LabelFrame(main_frame, text="Ввод панграммы", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.pangram_label = ttk.Label(
            input_frame,
            text=f'Введите: "{PANGRAM}"',
            font=(FONT_FAMILY, 11, 'bold'),
            foreground='darkblue'
        )
        self.pangram_label.pack(pady=(0, 5))
        
        self.typing_progress_label = ttk.Label(
            input_frame,
            text="",
            font=(FONT_FAMILY, 9, 'italic'),
            foreground='gray'
        )
        self.typing_progress_label.pack()
        
        self.text_entry = ttk.Entry(
            input_frame,
            width=60,
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.text_entry.pack(pady=5)
        
        self.status_label = ttk.Label(
            input_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.status_label.pack()
        
        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.submit_btn = ttk.Button(
            button_frame,
            text="Сохранить образец",
            command=self.submit_sample,
            state=tk.DISABLED
        )
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
        self.skip_btn = ttk.Button(
            button_frame,
            text="Пропустить фазу",
            command=self.skip_phase,
            state=tk.DISABLED
        )
        self.skip_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Отмена",
            command=self.window.destroy
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Настройка записи нажатий
        self.setup_keystroke_recording()
    
    def setup_keystroke_recording(self):
        """Настройка записи динамики нажатий"""
        self.text_entry.bind('<FocusIn>', self.start_recording)
        self.text_entry.bind('<FocusOut>', self.stop_recording)
        self.text_entry.bind('<KeyPress>', self.on_key_press)
        self.text_entry.bind('<KeyRelease>', self.on_key_release)
        self.text_entry.bind('<KeyRelease>', self.check_input, add='+')
        self.text_entry.bind('<Return>', lambda e: self.submit_sample())
    
    def start_testing(self):
        """Начало тестирования"""
        self.current_phase = "legitimate"
        self.samples_collected = 0
        self.update_interface()
        self.text_entry.focus()
    
    def update_interface(self):
        """Обновление интерфейса в зависимости от фазы"""
        if self.current_phase == "legitimate":
            self.phase_label.config(text="Фаза 1: Легитимные образцы")
            self.progress_label.config(text=f"Образцов собрано: {self.samples_collected}/{self.target_samples}")
            self.instructions_label.config(text=
                "Печатайте панграмму в ОБЫЧНОМ темпе, как при обучении системы. "
                "Это поможет получить образцы вашего естественного стиля печати."
            )
            self.skip_btn.config(state=tk.DISABLED)
            
        elif self.current_phase == "impostor_slow":
            self.phase_label.config(text="Фаза 2: Имитация (медленная печать)")
            self.progress_label.config(text=f"Образцов собрано: {self.samples_collected}/{self.target_samples//2}")
            self.instructions_label.config(text=
                "Печатайте панграмму МЕДЛЕННЕЕ обычного. Делайте паузы между словами, "
                "удерживайте клавиши дольше. Это имитирует попытку взлома."
            )
            self.skip_btn.config(state=tk.NORMAL)
            
        elif self.current_phase == "impostor_fast":
            self.phase_label.config(text="Фаза 3: Имитация (быстрая печать)")
            self.progress_label.config(text=f"Образцов собрано: {self.samples_collected}/{self.target_samples//2}")
            self.instructions_label.config(text=
                "Печатайте панграмму БЫСТРЕЕ обычного. Торопитесь, сокращайте паузы. "
                "Это также имитирует попытку взлома."
            )
            self.skip_btn.config(state=tk.NORMAL)
            
        elif self.current_phase == "completed":
            self.show_results()
            return
        
        # Обновление прогресс-бара
        if self.current_phase == "legitimate":
            max_val = self.target_samples
            current_val = self.samples_collected
        else:
            max_val = self.target_samples // 2
            current_val = self.samples_collected
            
        self.progress_bar.config(maximum=max_val, value=current_val)
    
    def start_recording(self, event=None):
        """Начало записи"""
        if not self.is_recording:
            self.session_id = self.keystroke_auth.start_keystroke_recording(self.user.id)
            self.is_recording = True
            self.status_label.config(text="Запись активна", foreground="red")
    
    def stop_recording(self, event=None):
        """Остановка записи"""
        if self.is_recording:
            self.is_recording = False
            self.status_label.config(text="Запись остановлена", foreground="gray")
    
    def on_key_press(self, event):
        """Обработка нажатия клавиши"""
        if self.is_recording and self.session_id:
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(self.session_id, event.keysym, 'press')
    
    def on_key_release(self, event):
        """Обработка отпускания клавиши"""
        if self.is_recording and self.session_id:
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(self.session_id, event.keysym, 'release')
    
    def check_input(self, event=None):
        """Проверка готовности ввода"""
        current_text = self.text_entry.get()
        normalized_current = self._normalize_text(current_text)
        
        # Проверяем длину
        if len(normalized_current) > len(self.normalized_target):
            self._reset_input("Текст слишком длинный")
            return
        
        # Проверяем правильность префикса
        is_correct_prefix = True
        for i, char in enumerate(normalized_current):
            if i >= len(self.normalized_target) or char != self.normalized_target[i]:
                is_correct_prefix = False
                break
        
        if not is_correct_prefix:
            self._reset_input("Ошибка в тексте")
            return
        
        # Обновляем прогресс ввода
        progress_text = f"Введено: {len(normalized_current)}/{len(self.normalized_target)} символов"
        self.typing_progress_label.config(text=progress_text)
        
        # Проверяем завершенность
        if normalized_current == self.normalized_target:
            self.submit_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Текст введен правильно", foreground="green")
        else:
            self.submit_btn.config(state=tk.DISABLED)
            if len(normalized_current) > 0:
                self.status_label.config(text="Продолжайте ввод", foreground="blue")
            else:
                self.status_label.config(text="Начните ввод панграммы", foreground="black")
    
    def _reset_input(self, message: str):
        """Сброс ввода при ошибке"""
        if self.is_recording:
            self.stop_recording()
            if self.session_id and self.session_id in self.keystroke_auth.current_session:
                del self.keystroke_auth.current_session[self.session_id]
            self.session_id = None
        
        self.text_entry.delete(0, tk.END)
        self.status_label.config(text=message, foreground="red")
        self.typing_progress_label.config(text="")
        self.submit_btn.config(state=tk.DISABLED)
        
        self.window.after(1500, self._clear_error_and_restart)
    
    def _clear_error_and_restart(self):
        """Очистка ошибки и перезапуск"""
        self.status_label.config(text="Начните ввод заново", foreground="black")
        self.text_entry.focus()
    
    def submit_sample(self):
        """Сохранение образца"""
        current_text = self.text_entry.get()
        normalized_current = self._normalize_text(current_text)
        
        if normalized_current != self.normalized_target:
            messagebox.showwarning("Предупреждение", "Введите панграмму полностью и правильно")
            return
        
        if self.session_id and self.is_recording:
            try:
                self.stop_recording()
                features = self.keystroke_auth.finish_recording(self.session_id, is_training=False)
                
                if not features or all(v == 0 for v in features.values()):
                    messagebox.showwarning("Предупреждение", 
                        "Не удалось записать динамику нажатий. Попробуйте печатать медленнее.")
                    self.text_entry.delete(0, tk.END)
                    self.text_entry.focus()
                    return
                
                # Сохраняем образец в соответствующую категорию
                sample_data = {
                    'features': features,
                    'timestamp': datetime.now(),
                    'phase': self.current_phase
                }
                
                if self.current_phase == "legitimate":
                    self.legitimate_samples.append(sample_data)
                else:
                    self.impostor_samples.append(sample_data)
                
                self.samples_collected += 1
                
                # Очищаем поле ввода
                self.text_entry.delete(0, tk.END)
                self.typing_progress_label.config(text="")
                self.status_label.config(text=f"Образец {self.samples_collected} сохранен", foreground="green")
                
                # Проверяем завершение фазы
                self.check_phase_completion()
                
                # Если тестирование не завершено, делаем паузу перед следующим образцом
                if self.current_phase != "completed":
                    self.text_entry.config(state=tk.DISABLED)
                    self.window.after(1000, self._enable_next_input)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")
                # Проверяем, что виджеты еще существуют перед обращением к ним
                try:
                    if self.window.winfo_exists():
                        self.text_entry.delete(0, tk.END)
                        self.text_entry.focus()
                except tk.TclError:
                    pass
    
    def _enable_next_input(self):
        """Разрешение следующего ввода"""
        # Проверяем, что окно еще существует
        try:
            if self.window.winfo_exists():
                self.text_entry.config(state=tk.NORMAL)
                self.text_entry.focus()
                self.status_label.config(text="", foreground="black")
        except tk.TclError:
            # Окно уже уничтожено, ничего не делаем
            pass
    
    def check_phase_completion(self):
        """Проверка завершения текущей фазы"""
        if self.current_phase == "legitimate" and self.samples_collected >= self.target_samples:
            self.next_phase()
        elif self.current_phase in ["impostor_slow", "impostor_fast"] and self.samples_collected >= self.target_samples // 2:
            self.next_phase()
    
    def next_phase(self):
        """Переход к следующей фазе"""
        if self.current_phase == "legitimate":
            self.current_phase = "impostor_slow"
            self.samples_collected = 0
            messagebox.showinfo("Переход к следующей фазе", 
                "Легитимные образцы собраны. Теперь будем имитировать попытки взлома.")
        elif self.current_phase == "impostor_slow":
            self.current_phase = "impostor_fast"
            self.samples_collected = 0
            messagebox.showinfo("Переход к следующей фазе", 
                "Медленная имитация завершена. Теперь попробуйте быструю печать.")
        elif self.current_phase == "impostor_fast":
            self.current_phase = "completed"
            messagebox.showinfo("Тестирование завершено", 
                "Все образцы собраны. Начинаем анализ эффективности системы.")
        
        self.update_interface()
    
    def skip_phase(self):
        """Пропуск текущей фазы"""
        if messagebox.askyesno("Подтверждение", "Пропустить текущую фазу тестирования?"):
            self.next_phase()
    
    def show_results(self):
        """Показ результатов тестирования - теперь в отдельном окне"""
        try:
            # Анализируем данные
            legitimate_features = [sample['features'] for sample in self.legitimate_samples]
            impostor_features = [sample['features'] for sample in self.impostor_samples]
            
            if not legitimate_features or not impostor_features:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для анализа")
                return
            
            # Тестируем модель на собранных данных
            results = self.calculate_metrics(legitimate_features, impostor_features)
            
            # Создаем отдельное окно для результатов
            ResultsWindow(self.parent, results, self.user)  # Используем self.parent вместо self.window
            
            # Закрываем окно тестирования
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")
    
    def calculate_metrics(self, legitimate_features: List[Dict], impostor_features: List[Dict]) -> Dict:
        """Расчет метрик эффективности"""
        all_features = legitimate_features + impostor_features
        all_labels = [1] * len(legitimate_features) + [0] * len(impostor_features)
        
        # Тестируем различные пороги
        thresholds = np.arange(0.1, 0.95, 0.05)
        metrics_results = []
        
        for threshold in thresholds:
            tp = 0  # True Positives (легитимные приняты)
            fp = 0  # False Positives (имитаторы приняты)
            tn = 0  # True Negatives (имитаторы отклонены)
            fn = 0  # False Negatives (легитимные отклонены)
            
            # Тестируем каждый образец
            for features, true_label in zip(all_features, all_labels):
                is_authenticated, confidence, _ = self.keystroke_auth.authenticate(self.user, features)
                predicted_label = 1 if confidence >= threshold else 0
                
                if true_label == 1 and predicted_label == 1:
                    tp += 1
                elif true_label == 0 and predicted_label == 1:
                    fp += 1
                elif true_label == 0 and predicted_label == 0:
                    tn += 1
                elif true_label == 1 and predicted_label == 0:
                    fn += 1
            
            # Расчет метрик
            far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
            frr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
            eer = (far + frr) / 2
            accuracy = ((tp + tn) / len(all_features)) * 100
            
            metrics_results.append({
                'threshold': threshold,
                'far': far,
                'frr': frr,
                'eer': eer,
                'accuracy': accuracy,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })
        
        # Находим оптимальные результаты
        optimal_result = min(metrics_results, key=lambda x: x['eer'])
        current_result = min(metrics_results, key=lambda x: abs(x['threshold'] - 0.75))
        
        # ROC данные
        all_confidences = []
        for features in all_features:
            _, confidence, _ = self.keystroke_auth.authenticate(self.user, features)
            all_confidences.append(confidence)
        
        return {
            'metrics_results': metrics_results,
            'optimal_result': optimal_result,
            'current_result': current_result,
            'all_confidences': all_confidences,
            'all_labels': all_labels,
            'legitimate_count': len(legitimate_features),
            'impostor_count': len(impostor_features)
        }


class ResultsWindow:
    """Отдельное окно для отображения результатов тестирования"""
    
    def __init__(self, parent, results: Dict, user: User):
        self.parent = parent
        self.results = results
        self.user = user
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Результаты контролируемого тестирования - {user.username}")
        self.window.geometry("1400x900")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_interface()
    
    def create_interface(self):
        """Создание интерфейса результатов"""
        # Заголовок
        header_frame = ttk.Frame(self.window, padding=10)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text=f"Результаты контролируемого тестирования - {self.user.username}",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack()
        
        # Основной контейнер с прокруткой
        main_canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязка колесика мыши
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Текстовые результаты
        text_frame = ttk.LabelFrame(scrollable_frame, text="Отчет", padding=10)
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.results_text = tk.Text(text_frame, height=15, width=120, font=(FONT_FAMILY, 9))
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=text_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Графики
        charts_frame = ttk.LabelFrame(scrollable_frame, text="Графики анализа", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.create_charts(charts_frame)
        
        # Кнопки
        buttons_frame = ttk.Frame(scrollable_frame, padding=10)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(buttons_frame, text="Сохранить отчет", 
                  command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Закрыть", 
                  command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Генерируем и показываем отчет
        report = self.generate_report()
        self.results_text.insert('1.0', report)
        self.results_text.config(state=tk.DISABLED)
    
    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        optimal = self.results['optimal_result']
        current = self.results['current_result']
        
        report = f"""РЕЗУЛЬТАТЫ КОНТРОЛИРУЕМОГО ТЕСТИРОВАНИЯ ЭФФЕКТИВНОСТИ

Пользователь: {self.user.username}
Дата тестирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}

Данные тестирования:
• Легитимных образцов: {self.results['legitimate_count']}
• Имитационных образцов: {self.results['impostor_count']}

Метрики при текущем пороге (75%):
• FAR (False Acceptance Rate): {current['far']:.2f}%
• FRR (False Rejection Rate): {current['frr']:.2f}%
• EER (Equal Error Rate): {current['eer']:.2f}%
• Общая точность: {current['accuracy']:.1f}%

Оптимальные метрики:
• Рекомендуемый порог: {optimal['threshold']:.0%}
• FAR при оптимальном пороге: {optimal['far']:.2f}%
• FRR при оптимальном пороге: {optimal['frr']:.2f}%
• EER при оптимальном пороге: {optimal['eer']:.2f}%

Confusion Matrix (текущий порог):
                Система ПРИНИМАЕТ    Система ОТКЛОНЯЕТ
Легитимный      TP: {current['tp']:8d}         FN: {current['fn']:8d}
Имитатор        FP: {current['fp']:8d}         TN: {current['tn']:8d}

Интерпретация результатов:
{self.interpret_results(current, optimal)}
"""
        return report
    
    def interpret_results(self, current: Dict, optimal: Dict) -> str:
        """Интерпретация результатов"""
        interpretations = []
        
        if current['eer'] < 10:
            interpretations.append("• Отличная система (EER < 10%)")
        elif current['eer'] < 20:
            interpretations.append("• Хорошая система (EER < 20%)")
        else:
            interpretations.append("• Система требует улучшения (EER >= 20%)")
        
        if current['far'] < 5:
            interpretations.append("• Высокая защита от имитаторов")
        elif current['far'] < 15:
            interpretations.append("• Приемлемая защита от имитаторов")
        else:
            interpretations.append("• Слабая защита от имитаторов")
        
        if current['frr'] < 15:
            interpretations.append("• Хорошее удобство для легитимного пользователя")
        elif current['frr'] < 30:
            interpretations.append("• Приемлемое удобство использования")
        else:
            interpretations.append("• Низкое удобство использования")
        
        return '\n'.join(interpretations)
    
    def create_charts(self, parent_frame):
        """Создание графиков"""
        # Создаем фигуру с графиками
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Результаты контролируемого тестирования', fontsize=14, fontweight='bold')
        
        # График 1: FAR vs FRR vs Порог
        thresholds = [r['threshold'] * 100 for r in self.results['metrics_results']]
        far_values = [r['far'] for r in self.results['metrics_results']]
        frr_values = [r['frr'] for r in self.results['metrics_results']]
        
        ax1.plot(thresholds, far_values, 'r-o', label='FAR', linewidth=2, markersize=4)
        ax1.plot(thresholds, frr_values, 'b-s', label='FRR', linewidth=2, markersize=4)
        ax1.axvline(75, color='gray', linestyle='--', alpha=0.7, label='Текущий порог')
        ax1.axvline(self.results['optimal_result']['threshold'] * 100, color='green', 
                   linestyle='--', alpha=0.7, label='Оптимальный порог')
        ax1.set_xlabel('Порог (%)')
        ax1.set_ylabel('Частота ошибок (%)')
        ax1.set_title('FAR и FRR vs Порог')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: EER vs Порог
        eer_values = [r['eer'] for r in self.results['metrics_results']]
        ax2.plot(thresholds, eer_values, 'g-^', label='EER', linewidth=3, markersize=6)
        ax2.axvline(75, color='gray', linestyle='--', alpha=0.7, label='Текущий порог')
        ax2.axvline(self.results['optimal_result']['threshold'] * 100, color='green', 
                   linestyle='--', alpha=0.7, label='Оптимальный порог')
        ax2.set_xlabel('Порог (%)')
        ax2.set_ylabel('EER (%)')
        ax2.set_title('Equal Error Rate vs Порог')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: ROC кривая
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(self.results['all_labels'], self.results['all_confidences'])
            roc_auc = auc(fpr, tpr)
            
            ax3.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC кривая (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Кривая')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except ImportError:
            ax3.text(0.5, 0.5, 'sklearn не доступен\nROC кривая недоступна', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # График 4: Распределение уверенности
        legitimate_confidences = self.results['all_confidences'][:self.results['legitimate_count']]
        impostor_confidences = self.results['all_confidences'][self.results['legitimate_count']:]
        
        ax4.hist(impostor_confidences, bins=15, alpha=0.7, color='red', 
                label=f'Имитаторы ({len(impostor_confidences)})', density=True, edgecolor='darkred')
        ax4.hist(legitimate_confidences, bins=15, alpha=0.7, color='green',
                label=f'Легитимные ({len(legitimate_confidences)})', density=True, edgecolor='darkgreen')
        ax4.axvline(0.75, color='black', linestyle='--', linewidth=2, label='Порог 75%')
        ax4.set_xlabel('Уверенность системы')
        ax4.set_ylabel('Плотность')
        ax4.set_title('Распределение уверенности')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Встраиваем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        canvas.draw()
    
    def save_report(self):
        """Сохранение отчета"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Сохранить отчет тестирования"
            )
            
            if filename:
                if filename.endswith('.json'):
                    # JSON отчет
                    report_data = {
                        'user': self.user.username,
                        'test_date': datetime.now().isoformat(),
                        'legitimate_samples': self.results['legitimate_count'],
                        'impostor_samples': self.results['impostor_count'],
                        'current_metrics': self.results['current_result'],
                        'optimal_metrics': self.results['optimal_result'],
                        'all_metrics': self.results['metrics_results']
                    }
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                else:
                    # Текстовый отчет
                    report = self.generate_report()
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)
                
                messagebox.showinfo("Успех", f"Отчет сохранен: {filename}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")