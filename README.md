# Diabetes Prediction — Logistic Regression | KNN | Naive Bayes

A machine learning project that trains and compares three classifiers on the **Pima Indians Diabetes Database** to predict whether a patient has diabetes.

---

## Project Structure

```
├── Diabetes-Prediction.PY       # Main script — training, evaluation, visualisations
├── download_data(1).py          # Optional: download dataset via OpenML API
├── diabetes.csv                 # Dataset (768 samples, 9 features)
├── diabetes_model_analysis.png  # 9-panel model dashboard (saved output)
├── knn_k_sweep.png              # KNN k-value sweep plot (saved output)
└── requirements.txt             # (install dependencies — see below)
```

---

## Dataset

**Pima Indians Diabetes Database** — originally from the UCI Machine Learning Repository, also available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hour OGTT) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree score |
| Age | Age in years |
| **Outcome** | **Target — 1 = Diabetes, 0 = No Diabetes** |

- 768 total samples | 500 negative (65%) / 268 positive (35%)

---

## Models

| Model | Notes |
|---|---|
| Logistic Regression | L2 regularised, `max_iter=1000` |
| K-Nearest Neighbors | Default `k=5`; best k found via sweep (k=1–20) |
| Gaussian Naive Bayes | Assumes feature independence |

All models are evaluated **with and without StandardScaler** to demonstrate the effect of feature scaling.

---

## Results (With Scaling)

| Model | Test Accuracy | ROC-AUC | Overfit Gap |
|---|---|---|---|
| **KNN** | **0.7532** | 0.7886 | 0.0774 |
| Logistic Regression | 0.7078 | **0.8130** | 0.0886 |
| Naive Bayes | 0.7013 | 0.7646 | 0.0625 |

- **Best accuracy:** KNN (k=8) — 75.3%
- **Best ROC-AUC:** Logistic Regression — 0.813
- **Least overfit:** Naive Bayes — gap of 0.0625

---

## Key Observations

- **Feature scaling is critical for KNN.** Without scaling, KNN accuracy drops from 75% to 67% because Euclidean distances are dominated by high-range features like Insulin and Glucose.
- **Logistic Regression** has the best ROC-AUC, meaning it ranks positive cases most reliably even if raw accuracy is slightly lower.
- **Naive Bayes** is scale-invariant and surprisingly competitive despite assuming feature independence.
- **KNN with small k** tends to overfit — training accuracy approaches 100% while test accuracy drops.

---

## Visualisations

The script produces two saved plots:

**`diabetes_model_analysis.png`** — 9-panel dashboard:
- Test accuracy: scaled vs unscaled
- Train vs test accuracy (overfitting view)
- ROC-AUC comparison
- Confusion matrices for each model
- ROC curves overlay
- Overfitting gap bar chart
- Feature importance (Logistic Regression coefficients)

**`knn_k_sweep.png`** — Train and test accuracy for k = 1 to 20, with the best k highlighted.

---

## Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/shakib79ai/Logistic-Regression-KNN-Naive-Bayes.git
cd Logistic-Regression-KNN-Naive-Bayes
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4. Run the main script

```bash
python Diabetes-Prediction.PY
```

The script loads `diabetes.csv` from the same directory, trains all models, prints results to the console, and saves both plots.

### (Optional) Download dataset via OpenML

```bash
pip install openml
python "download_data(1).py"
```

---

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

---

## License

This project is for educational purposes. The dataset is publicly available under the UCI Machine Learning Repository terms.
