#  Data Classification Using AI — KNN on Iris Dataset

A supervised machine learning pipeline that classifies iris flowers into 3 species using the K-Nearest Neighbors algorithm. Built as **Project 2** of the DecodeLabs AI Industrial Training Kit (Batch 2026).

---

##  Results

| Metric | Score |
|---|---|
| Accuracy | **100%** |
| F1 Score (weighted) | **1.0000** |
| Misclassifications | **0 / 30** |

---

##  What It Does

Implements a full ML pipeline following the **IPO Framework**:

```
INPUT          →        PROCESS         →       OUTPUT
──────────────────────────────────────────────────────
Iris Dataset        Train-Test Split         Confusion Matrix
Feature Scaling     KNN Algorithm (k=5)      F1 Score
                    model.fit()              Classification Report
                    model.predict()          Live Prediction
```

---

##  The Dataset: Iris Benchmark

| Property | Value |
|---|---|
| Samples | 150 (balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |

The Iris dataset is built into scikit-learn — no download needed.

---

##  How It Works

**Step 1 — Load & Explore**
```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```

**Step 2 — Feature Scaling (The Gatekeeper Rule)**
KNN uses Euclidean distance. Without scaling, large-range features dominate unfairly.
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Result: mean=0, variance=1 for all features
```

**Step 3 — Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
# 120 training samples | 30 testing samples
```

**Step 4 — KNN Algorithm (k=5)**
```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)        # FIT: memorize the map
predictions = model.predict(X_test) # PREDICT: apply logic
```

**Step 5 — Validate with F1 + Confusion Matrix**
```python
f1_score(y_test, predictions, average='weighted')
confusion_matrix(y_test, predictions)
```

---

##  How to Run

**Install dependencies:**
```bash
pip install scikit-learn pandas numpy
```

**Run:**
```bash
python classifier.py
```

**Expected output:**
```
 Accuracy  : 100.00%
 F1 Score  : 1.0000
 Predicted Species : SETOSA
```

---

##  Project Structure

```
 iris-knn-classifier
 ┗  classifier.py    # Full ML pipeline
 ┗  README.md        # Documentation
```

---

##  Key Concepts Demonstrated

| Concept | What It Means |
|---|---|
| Supervised Learning | Model learns from labeled examples |
| Feature Scaling | Prevents distance bias in KNN |
| Train-Test Split | Tests model on unseen data (no cheating) |
| KNN Algorithm | Classifies by majority vote of k nearest neighbors |
| Confusion Matrix | Breaks down TP, FP, FN, TN per class |
| F1 Score | Harmonic mean of precision & recall — better than raw accuracy |

---

##  Why K=5?

- **K=1** → Overfitting: memorizes noise, fails on new data
- **K=5** → The "elbow" sweet spot: balanced precision
- **K=100** → Underfitting: too generic, misses real patterns

---

##  About

Built as part of the **DecodeLabs AI Industrial Training Programme**.
Project 2 bridges rule-based logic (Project 1) to full supervised learning pipelines.
