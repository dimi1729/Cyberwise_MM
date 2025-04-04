import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from analysis.plotting import plot_confusion_matrix, plot_betas

df = pd.read_csv('data/even_merged_1-10.csv')


df["Malicious"] = df["Label"] != "BENIGN"
X = df.drop(columns=['Malicious', 'Label'])  
y = df['Malicious']  

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)




print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    ["Benign", "Malicious"], 
    "knn"
)