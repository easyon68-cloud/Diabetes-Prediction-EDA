# 🩺 Diabetes Prediction — EDA & Machine Learning

> Can we predict whether a person has diabetes just from basic health measurements — like glucose level, BMI, and age? This project explores the famous **Pima Indians Diabetes Dataset** using deep visual analysis and compares multiple machine learning models to find the most effective one.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 📌 Project Overview

Doctors and researchers often want to catch diabetes risk early, before it becomes a serious health problem. This project simulates that process using data:

1. **Clean** messy medical data (fixing impossible "zero" values that sneak into real-world datasets)
2. **Explore** the data visually to understand what separates diabetic patients from non-diabetic ones
3. **Train and compare 5 machine learning models** to find the one that predicts diabetes most reliably
4. **Evaluate** the winning model in depth — not just accuracy, but precision, recall, and AUC

By the end, you'll know exactly which health factors matter most for diabetes risk, and you'll have a tuned **Random Forest model reaching ~78% cross-validated accuracy** with a strong 0.82 AUC score.

---

## 📊 Dataset

The dataset contains health records for **768 patients**, with the following features:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index — weight relative to height |
| `DiabetesPedigreeFunction` | A score estimating diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | **Target** — 1 = Diabetic, 0 = Not Diabetic |

📁 File expected at: `diabetes.csv`

**Why this dataset is tricky:** all patients are female, at least 21 years old, and of Pima Indian heritage — a population with a historically high rate of type 2 diabetes. That makes it a great case study, but also means conclusions apply specifically to this group, not the general population.

---

## 🧹 Data Cleaning

Several columns contained `0` values that are **medically impossible** — a living patient cannot have 0 Glucose or 0 BMI. These are actually **missing data in disguise**.

**What we did:**
1. Identified the affected columns: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
2. Replaced every `0` in those columns with `NaN`
3. Filled missing values using each column's **median**, which resists distortion from outliers (this dataset has plenty, especially in Insulin)

This step matters — feeding a model "0 BMI" would quietly corrupt predictions without ever throwing an error.

---

## 🔍 Exploratory Data Analysis (EDA)

### 1️⃣ Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png) ![Uploading image.png…]()


`Glucose` has the strongest correlation with `Outcome`, followed by `BMI` and `Age` — our first clue about which features will drive the model.

### 2️⃣ Class Balance
![Outcome Countplot](images/outcome_countplot.png)

The dataset is **slightly imbalanced** (500 non-diabetic vs 268 diabetic). This is why the models below use `class_weight='balanced'` — so the model doesn't just lazily favor the majority class.

### 3️⃣ Boxplots: Glucose, BMI & Age by Outcome
![Boxplots of Key Features](images/boxplots_key_features.png)

Diabetic patients show a **visibly higher median Glucose**, moderately higher BMI, and a gentle upward shift in Age. This chart basically previews the final model's conclusions.

### 4️⃣ Distribution of All Numerical Features
![Feature Histograms](images/feature_histograms.png)

`Glucose` and `BMI` are close to a normal (bell-curve) distribution — good for modeling. `Insulin` and `SkinThickness` are heavily right-skewed with extreme outliers.

### 5️⃣ Diabetes Prevalence by Age Group
![Age Group vs Outcome](images/age_group_outcome.png)

Diabetes becomes **more common with age**, especially from the 30s onward — consistent with real-world medical patterns for type 2 diabetes.

### 6️⃣ Glucose vs BMI — Combined Risk Zone
![Glucose vs BMI Joint Plot](images/glucose_bmi_joint.png)

Diabetic patients cluster in the **high-Glucose, high-BMI region**. The combination of both factors is a stronger signal than either alone.

---

## 🤖 Model Comparison — Finding the Best Algorithm

Instead of relying on a single model, **5 different algorithms** were trained and evaluated using **5-fold cross-validation** for a robust, non-lucky-split comparison:

![Model Comparison](images/model_comparison.png)

| Model | CV Accuracy |
|---|---|
| 🥇 **Random Forest** | **77.8%** |
| 🥈 KNN | 77.0% |
| 🥉 Logistic Regression | 76.2% |
| SVM (RBF) | 76.2% |
| Gradient Boosting | 74.9% |

**Random Forest came out on top** — it handles the non-linear relationships between features (like the Glucose × BMI risk zone we saw earlier) better than a straight-line model like Logistic Regression can.

---

## 🏆 Final Model: Random Forest (Tuned)

| Setting | Value |
|---|---|
| Algorithm | `RandomForestClassifier` |
| Trees | 400 |
| Max Depth | 6 |
| Class Weighting | Balanced (compensates for the imbalanced dataset) |
| Validation | 5-fold Stratified Cross-Validation + held-out 20% test set |

### 📈 ROC Curve — Random Forest vs Logistic Regression
![ROC Curve](images/roc_curve.png)

The ROC curve shows how well each model separates diabetic from non-diabetic patients across every possible decision threshold — the further the curve bows toward the top-left corner, the better. **Random Forest edges out Logistic Regression (AUC 0.824 vs 0.813).**

### 🧮 Confusion Matrix — Final Model
![Confusion Matrix RF](images/confusion_matrix_rf.png)

| Metric | Score | What it means |
|---|---|---|
| **Accuracy** | **75.3%** | Correctly classifies 3 out of 4 patients |
| **Precision** | 62.9% | When it predicts "diabetic," it's right 63% of the time |
| **Recall** | **72.2%** | It correctly catches 72% of all real diabetic patients |
| **F1-score** | 67.2% | Balanced measure of precision and recall |
| **ROC-AUC** | **0.824** | Strong ability to distinguish diabetic vs non-diabetic patients |

> 💡 **Why this matters:** compared to a plain Logistic Regression baseline (62% recall), the tuned Random Forest catches **10 percentage points more actual diabetic patients** — reducing dangerous false negatives, which matters most in a medical screening context.

### 🔑 Feature Importance — Random Forest
![Feature Importance RF](images/feature_importance_rf.png)

Random Forest confirms what the EDA hinted at: **Glucose is by far the strongest predictor**, followed by **BMI**, **Age**, and **Insulin**.

---

## 📝 Summary of Insights

| Insight | Why it matters |
|---|---|
| Glucose is the #1 predictor across every model and every chart | Most reliable single signal for diabetes risk |
| BMI and Age are strong secondary predictors | Support Glucose as compounding risk factors |
| High Glucose + high BMI together = highest risk zone | Risk factors compound, not just add |
| Random Forest outperforms linear models | Diabetes risk isn't purely linear — trees capture interactions better |
| Class balancing improves recall significantly | Fewer missed diabetic patients — critical for a health screening tool |
| Dataset is slightly imbalanced | Accuracy alone is misleading — precision/recall/AUC give the full picture |

---

## 🛠️ Tech Stack

- **Python 3**
- `pandas`, `numpy` — data loading and cleaning
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, cross-validation, ROC/AUC)

---

## 🚀 How to Run

1. Clone/download this repository
2. Make sure `diabetes.csv` is available in the working directory
3. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. Open and run `diabetes_dataset.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab

---

## 📈 Possible Next Steps

- Try SMOTE for synthetic oversampling of the minority (diabetic) class
- Add feature interactions (e.g. Glucose × BMI) directly into the model
- Explore SHAP values for deeper per-patient explainability
- Deploy the trained Random Forest as a simple web app for real-time risk prediction

---

## 📄 License

This project is for educational and research purposes, using the publicly available Pima Indians Diabetes Dataset.

---

⭐ *If you found this project useful, consider giving it a star!*
