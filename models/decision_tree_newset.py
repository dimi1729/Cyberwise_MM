import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

train_csv = '../data/UNSW_NB15_training-set(in).csv'
test_csv = '../data/UNSW_NB15_testing-set(in).csv'

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

merged_data = pd.concat([train_data, test_data], ignore_index=True)

X = merged_data.drop(columns=['label', 'proto', 'service', 'state', 'attack_cat'])
y = merged_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

plot_confusion_matrix(
    model, X_test_scaled, y_test,
    display_labels=["Benign", "Malicious"],
    cmap=plt.cm.Blues
)
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("Feature Importance (sorted by importance):")
print(feature_importance[['Feature', 'Importance']])
