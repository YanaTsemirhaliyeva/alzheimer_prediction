#!/usr/bin/env python
# coding: utf-8

from data_loader import DataProcessor
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Шаг 1: Загрузка данных
    processor = DataProcessor()
    processor.load_csv('data/alzheimers_prediction_dataset.csv')
    
    # Шаг 2: Анализ данных
    print("\nАнализ пропущенных значений:")
    processor.count_missing_values()
    
    print("\nЗаполнение пропущенных значений:")
    processor.fill_missing_values(strategy='mean')
    
    print("\nПроверка значимости факторов с помощью хи-квадрат теста:")
    for factor in processor.data.columns:
        if factor != 'Alzheimer’s Diagnosis':
            try:
                contingency_table = pd.crosstab(processor.data[factor], processor.data["Alzheimer’s Diagnosis"])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(f"Фактор: {factor}")
                print(f"Хи-квадрат: {chi2:.2f}, p-значение: {p:.4f}")
                if p < 0.05:
                    print("=> Значимая связь с диагнозом болезни Альцгеймера\n")
                else:
                    print("=> Нет значимой связи\n")
            except Exception as e:
                print(f"Ошибка при анализе фактора {factor}: {e}")
    
    # Шаг 3: Визуализация данных
    # visualizer = DataVisualizer(processor.data)
    # print("\nПостроение визуализаций:")
    # visualizer.plot_histogram('Age')
    # visualizer.plot_line('Age', 'Cognitive_Score')
    # visualizer.plot_scatter('Alzheimer’s Diagnosis')
    
    # Шаг 4: Подготовка данных для предсказания
    print("\nПодготовка данных для обучения моделей...")
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
    
    # Шаг 5: Логистическая регрессия
    print("\nОбучение логистической регрессии...")
    logistic_model = LogisticRegression(C=0.1, max_iter=100, solver='newton-cg')
    logistic_model.fit(scaled_X_train, y_train)
    y_pred = logistic_model.predict(scaled_X_test)
    
    print("\nРезультаты логистической регрессии:")
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    ConfusionMatrixDisplay.from_estimator(logistic_model, scaled_X_test, y_test)
    plt.show()
    
    # Кросс-валидация
    print("\nКросс-валидация логистической регрессии:")
    scores = cross_val_score(logistic_model, scaled_X_train, y_train, cv=5, scoring='accuracy')
    print(f'Средняя точность: {scores.mean() * 100:.2f}%')
    
    # Шаг 6: RandomForestClassifier
    print("\nОбучение RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        max_depth=20,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print("\nРезультаты RandomForestClassifier:")
    print(classification_report(y_test, y_pred_rf))
    print(f'Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%')
    ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)
    plt.show()

if __name__ == "__main__":
    main()