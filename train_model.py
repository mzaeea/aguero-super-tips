import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/historical_matches.csv')

features = df[['feature1', 'feature2']]
target = df['target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'ai/model.pkl')

print("✅ تم تدريب النموذج وحفظه بنجاح")
