#!/usr/bin/env python
# coding: utf-8

# In[27]:


from data_loader import DataProcessor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# In[28]:


# Подготовка данных
processor = DataProcessor()
processor.load_csv('data/alzheimers_prediction_dataset.csv')

X = processor.data.drop('Alzheimer’s Diagnosis', axis=1)
X = pd.get_dummies(X)
y = processor.data['Alzheimer’s Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
scaled_X_train = imputer.fit_transform(scaled_X_train)
scaled_X_test = imputer.transform(scaled_X_test)


# In[29]:


# Логистическая регрессия
model = LogisticRegression(C=0.1, max_iter=100, solver='newton-cg')
model.fit(scaled_X_train, y_train)

# Предсказание
y_pred = model.predict(scaled_X_test)

# Оценка модели
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
ConfusionMatrixDisplay.from_estimator(model, scaled_X_test, y_test)


# In[30]:


# попробуем кросс-валидацию
scores = cross_val_score(model, scaled_X_train, y_train, cv=5, scoring='accuracy')
print(f'Средняя точность модели: {scores.mean() * 100:.2f}%')


# In[31]:


# проверяю дисбаланс классов
print(y.value_counts())


# In[32]:


# устраняю дисбаланс балансировкой
model = LogisticRegression(class_weight='balanced', C=0.1, max_iter=100, solver='newton-cg')
model.fit(scaled_X_train, y_train)

# Предсказание
y_pred = model.predict(scaled_X_test)

# Оценка модели
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
ConfusionMatrixDisplay.from_estimator(model, scaled_X_test, y_test)


# In[33]:


# RandomForestClassifier
model = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# In[26]:


# Учитываем дисбаланс классов
model = RandomForestClassifier(
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=200,
    random_state=1342,
    class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);

