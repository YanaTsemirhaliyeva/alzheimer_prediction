#!/usr/bin/env python
# coding: utf-8

# In[28]:


from data_loader import DataProcessor
import pandas as pd
from scipy.stats import chi2_contingency


# In[29]:


processor = DataProcessor()
processor.load_csv('data/alzheimers_prediction_dataset.csv')

processor.count_missing_values()


# In[30]:


processor.fill_missing_values() 


# In[31]:


# Проверка значимости для каждого фактора
for factor in processor.data.columns:
    if factor != 'Alzheimer’s Diagnosis':
        contingency_table = pd.crosstab(processor.data[factor], processor.data["Alzheimer’s Diagnosis"])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
    
        print(f"Фактор: {factor}")
        print(f"Хи-квадрат: {chi2:.2f}, p-значение: {p:.4f}")
        if p < 0.05:
            print("=> Значимая связь с диагнозом болезни Альцгеймера\n")
        else:
            print("=> Нет значимой связи\n")


# In[ ]:




