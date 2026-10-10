import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('d:/Machine Learning/ML Practice/diabetes_dataset.csv')
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
base = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)
base.fit(X_train, y_train)
print('base acc', accuracy_score(y_test, base.predict(X_test)))
param_grid = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced'],
    'criterion': ['gini', 'entropy'],
}
search = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
search.fit(X_train, y_train)
print('best params', search.best_params_)
print('best cv', search.best_score_)
print('best test', accuracy_score(y_test, search.best_estimator_.predict(X_test)))
