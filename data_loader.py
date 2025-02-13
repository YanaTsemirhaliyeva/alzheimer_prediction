#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


class DataProcessor:
    def __init__(self):
        self.data = None

    def load_csv(self, path: str) -> None:
        try:
            self.data = pd.read_csv(path)
            print('Данные успешно загружены из csv!')
        except Exception as e:
            print(f'Ошибка при загрузке csv: {e}')

    def watch_data_head(self) -> None:
        if self.data is not None:
            print(self.data.head())
        else:
            print('Данные не загружены. Используйте метод load_csv()')

    def count_missing_values(self) -> None:
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print(f'Количество пропущенных значений в каждом столбце: \n{missing_values}')
        else:
            print('Данные не загружены. Используйте метод load_csv() для загрузки данных.')

    def fill_missing_values(self, strategy: str = 'mean') -> None:
        if self.data is not None:
        # Проверяем, есть ли пропущенные значения
            if self.data.isnull().sum().sum() == 0:
                print("Нет пропущенных значений. Заполнение не требуется.")
                return
    
            # Выполняем заполнение в зависимости от выбранной стратегии
            try:
                if strategy == 'mean':
                    self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                elif strategy == 'median':
                    self.data.fillna(self.data.median(numeric_only=True), inplace=True)
                elif strategy == 'most_frequent':
                    self.data.fillna(self.data.mode().iloc[0], inplace=True)
                else:
                    print("Неподдерживаемая стратегия заполнения. Используйте 'mean', 'median' или 'most_frequent'.")
                    return
                print(f"Пропущенные значения успешно заполнены с использованием стратегии: {strategy}.")
            except Exception as e:
                print(f"Произошла ошибка при заполнении значений: {e}")
        else:
            print("Данные не загружены. Используйте метод load_csv() для загрузки данных.")


# In[ ]:




