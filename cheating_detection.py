import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("exam_data.csv")
print(data.head())

print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())


sns.countplot(x='cheating', data=data)
plt.title("Cheating vs Non-Cheating")
plt.show()

sns.boxplot(x='cheating', y='tab_switch_count', data=data)
plt.title("Tab Switching vs Cheating")
plt.show()

X = data.drop(['student_id', 'cheating'], axis=1)
y = data['cheating']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importance in Cheating Detection")
plt.show()

import joblib
joblib.dump(model, "cheating_model.pkl")
print("Model saved successfully")


# exam_duration, time_per_question, tab_switch_count, same_answer_ratio, ip_change
new_student = [[60, 20, 14, 0.82, 1]]

result = model.predict(new_student)

if result[0] == 1:
    print("🚨 Cheating Detected")
else:
    print("✅ No Cheating")
