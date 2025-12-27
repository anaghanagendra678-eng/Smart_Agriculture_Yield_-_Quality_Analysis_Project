# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ===============================
# 2. LOAD & FIX DATASET
# ===============================
# Load CSV (robust for formatting issues)
raw_df = pd.read_csv("crop_recommendation.csv", header=0)

# If dataset is read as a single column, split it
if len(raw_df.columns) == 1:
    df = raw_df[raw_df.columns[0]].str.split(",", expand=True)
    df.columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col])
else:
    df = raw_df.copy()

print("Dataset Loaded Successfully")
print(df.head())
print(df.columns)

# ===============================
# 3. CREATE QUALITY LABEL
# ===============================
def quality_label(rainfall):
    if rainfall > 200:
        return "Good"
    elif rainfall > 100:
        return "Average"
    else:
        return "Poor"

df['Quality'] = df['rainfall'].apply(quality_label)

# ===============================
# 4. FEATURE & TARGET SPLIT
# ===============================
X = df.drop(['label', 'Quality'], axis=1)
y = df['Quality']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# ===============================
# 5. TRAIN–TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. FEATURE SCALING (IMPORTANT FOR KNN)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 7. TRAIN KNN MODEL
# ===============================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ===============================
# 8. PREDICTION
# ===============================
y_pred = knn.predict(X_test)

# ===============================
# 9. EVALUATION METRICS
# ===============================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ===============================
# 10. CONFUSION MATRIX VISUALIZATION
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()

# ===============================
# 11. ACCURACY vs K VALUE GRAPH
# ===============================
accuracy_list = []

for k in range(1, 21):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    pred_temp = knn_temp.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, pred_temp))

plt.figure(figsize=(7,5))
plt.plot(range(1, 21), accuracy_list, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K Value")
plt.grid(True)
plt.show()

# ===============================
# 12. FEATURE VISUALIZATION
# ===============================
plt.figure(figsize=(6,5))
sns.boxplot(x='Quality', y='rainfall', data=df)
plt.title("Rainfall vs Crop Quality")
plt.show()

# ===============================
# 13. SAMPLE PREDICTION
# ===============================
sample = [[90, 42, 43, 25, 80, 6.5, 150]]  # N, P, K, temp, humidity, ph, rainfall
sample_scaled = scaler.transform(sample)
prediction = knn.predict(sample_scaled)
print("Predicted Crop Quality:", le.inverse_transform(prediction)[0])
