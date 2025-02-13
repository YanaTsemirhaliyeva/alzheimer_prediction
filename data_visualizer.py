#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, column: str) -> None:
        if self.data is not None:
            if column in self.data.columns:
                self.data[column].hist(bins=20)
                plt.title(f'Гистограмма: {column}')
                plt.xlabel(column)
                plt.ylabel('Частота')
                plt.show()
            else:
                print(f'Столбец {column} не найден в данных')
        else:
            print('Данные отсутствуют')

    def plot_line(self, column_x: str, column_y: str) -> None:
        if self.data is not None:
            if column_x in self.data.columns and column_y in self.data.columns:
                # Ограничиваем данные для построения графика (каждая 100-я строка)
                subset = self.data.iloc[::500]

                # Данные для осей
                x_values = range(len(subset))
                y_true = subset[column_x]
                y_pred = subset[column_y]

                # Настройка графика
                plt.figure(figsize=(12, 6))
                plt.plot(x_values, y_true, label=column_x, marker='o', linestyle='-', color='blue', alpha=0.7)
                plt.plot(x_values, y_pred, label=column_y, marker='x', linestyle='--', color='orange', alpha=0.7)

                # Оформление
                plt.xlabel('Наблюдения')
                plt.ylabel('Значения')
                plt.title(f'Сравнение: {column_x} vs {column_y}')
                plt.legend()
                plt.grid(True)

                # Отображение графика
                plt.show()
            else:
                 print(f"Столбцы '{column_x}' или '{column_y}' не найдены в данных.")
        else:
            print('Данные отсутствуют.')

    def plot_scatter(self, hue_column) -> None:
            if not hue_column in self.data.columns:
                print('Данные по признаку отсутствуют')
                return
                
            if self.data is not None:
                # Построение диаграмм рассеивания для всех признаков
                plt.figure(figsize=(12, 8))
                sns.pairplot(
                    self.data, 
                    hue=hue_column,  # Цвет точек в зависимости от метки класса
                    diag_kind='kde',  # Диагональные графики - оценки плотности
                    palette='viridis', 
                    plot_kws={'alpha': 0.7}  # Прозрачность точек
                )
                plt.suptitle("Диаграммы рассеивания для всех признаков", y=1.02)
                plt.show()
            else:
                print("Данные отсутствуют.")

