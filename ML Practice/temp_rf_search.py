import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('d:/Machine Learning/ML Practice/diabetes_dataset.csv')
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 8, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7, 1.0],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=80,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
search.fit(X_train, y_train)
print('best params', search.best_params_)
print('best cv', round(search.best_score_, 4))
print('test acc', round(accuracy_score(y_test, search.best_estimator_.predict(X_test)), 4))
