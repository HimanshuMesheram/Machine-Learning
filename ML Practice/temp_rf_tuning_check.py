import pandas as pd
from sklearn.model_selection import train_test_split
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

candidates = [
    dict(n_estimators=200, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', criterion='gini'),
    dict(n_estimators=200, max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', criterion='gini'),
    dict(n_estimators=400, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', criterion='entropy'),
    dict(n_estimators=300, max_depth=7, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', criterion='entropy'),
    dict(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', criterion='entropy'),
    dict(n_estimators=200, max_depth=5, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', criterion='gini'),
    dict(n_estimators=200, max_depth=9, min_samples_split=2, min_samples_leaf=1, max_features='log2', class_weight='balanced', criterion='entropy'),
    dict(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight=None, criterion='gini'),
]

for params in candidates:
    model = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True, **params)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(params, '-> acc', round(acc, 4), 'oob', round(model.oob_score_, 4))
