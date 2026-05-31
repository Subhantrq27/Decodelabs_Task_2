# ============================================================
#  DecodeLabs | Batch 2026 | Project 2: Data Classification
#  Pipeline: Iris Dataset -> KNN -> Confusion Matrix + F1 Score
#  IPO Framework: Input -> Process -> Output
# ============================================================

# -- DEPENDENCIES --------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import pandas as pd

# ============================================================
# PHASE 1 -- INPUT: Load & Understand the Dataset
# ============================================================

print("=" * 60)
print("  DecodeLabs | Project 2 | Data Classification")
print("=" * 60)

# Load the Iris benchmark dataset
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

# Explore the dataset
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in y]

print("\nDATASET OVERVIEW")
print(f"   Samples   : {X.shape[0]}")
print(f"   Features  : {X.shape[1]} -> {list(iris.feature_names)}")
print(f"   Classes   : {len(iris.target_names)} -> {list(iris.target_names)}")
print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))

# -- GATEKEEPER RULE: Feature Scaling ------------------------
# KNN uses distance -- without scaling, larger-range features dominate.
# StandardScaler normalizes: mean=0, variance=1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[OK] Feature scaling applied (StandardScaler: mean=0, variance=1)")

# ============================================================
# PHASE 2 -- PROCESS: Train-Test Split + KNN Algorithm
# ============================================================

# Split: 80% training, 20% testing | shuffle=True removes order bias
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTRAIN-TEST SPLIT")
print(f"   Training samples : {len(X_train)} (80%)")
print(f"   Testing samples  : {len(X_test)}  (20%)")

# Instantiate the KNN model (K=5 -- the elbow sweet spot)
# K=1 -> overfitting (noise), K=100 -> underfitting (too generic)
model = KNeighborsClassifier(n_neighbors=5)

# FIT -- model memorizes the training map
model.fit(X_train, y_train)
print("\nModel trained: KNeighborsClassifier(k=5)")

# PREDICT -- model applies learned logic to unseen test data
predictions = model.predict(X_test)

# ============================================================
# PHASE 3 -- OUTPUT: Validation (Confusion Matrix + F1 Score)
# ============================================================

print("\n" + "=" * 60)
print("  OUTPUT VALIDATION")
print("=" * 60)

# -- Accuracy (the basic metric -- can be misleading on imbalanced data)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy  : {accuracy * 100:.2f}%")

# -- F1 Score (the honest metric -- harmonic mean of precision & recall)
f1 = f1_score(y_test, predictions, average="weighted")
print(f"F1 Score  : {f1:.4f}  (weighted average)")

# -- Confusion Matrix -- the full diagnostic tool
cm = confusion_matrix(y_test, predictions)
print("\nCONFUSION MATRIX")
print(f"   Rows = Actual Class | Columns = Predicted Class")
print(f"   Classes: {list(iris.target_names)}\n")
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {n}" for n in iris.target_names],
    columns=[f"Pred: {n}" for n in iris.target_names],
)
print(cm_df.to_string())

# -- Classification Report -- precision, recall, F1 per class
print("\nFULL CLASSIFICATION REPORT")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# -- Misclassification summary
misclassified = (predictions != y_test).sum()
print(f"Misclassified samples : {misclassified} / {len(y_test)}")
print(f"Correctly classified  : {len(y_test) - misclassified} / {len(y_test)}")

# ============================================================
# BONUS: Predict a custom sample
# ============================================================
print("\n" + "=" * 60)
print("  LIVE PREDICTION DEMO")
print("=" * 60)
print("  Custom sample: sepal_length=5.1, sepal_width=3.5,")
print("                 petal_length=1.4, petal_width=0.2\n")

sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"  Predicted Species : {iris.target_names[prediction[0]].upper()}")

print("\n" + "=" * 60)
print("  Pipeline complete. Model ready for deployment.")
print("=" * 60)
