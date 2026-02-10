# Diabetes-Prediction-EDA
# 🩺 Diabetes Prediction — Exploratory Data Analysis (EDA)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **A comprehensive exploratory data analysis on the Pima Indians Diabetes Dataset to uncover patterns, clean data, and build a predictive model for early diabetes detection.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [EDA Workflow](#-eda-workflow)
- [Visualizations](#-visualizations)
- [Predictive Modeling](#-predictive-modeling)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🔍 Overview

Diabetes is one of the fastest-growing chronic diseases worldwide, affecting over **500 million people** globally. Early detection through data-driven insights can drastically improve patient outcomes and reduce healthcare costs.

This project performs a **deep Exploratory Data Analysis (EDA)** on the Pima Indians Diabetes Dataset to:

- Understand the distribution and relationships of health-related features
- Identify and handle data quality issues (zero-value imputation)
- Uncover the most influential predictors of diabetes
- Build a baseline predictive model using **Logistic Regression**

---

## 📊 Dataset Description

The dataset is sourced from the **National Institute of Diabetes and Digestive and Kidney Diseases** and includes diagnostic data from **768 female patients** of Pima Indian heritage.

| Feature | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of times pregnant | Count |
| `Glucose` | Plasma glucose concentration (2-hr oral test) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin fold thickness | mm |
| `Insulin` | 2-Hour serum insulin | mu U/ml |
| `BMI` | Body Mass Index | kg/m² |
| `DiabetesPedigreeFunction` | Family history-based diabetes likelihood score | Score |
| `Age` | Age of the patient | Years |
| `Outcome` | **Target variable** — 1 = Diabetic, 0 = Non-Diabetic | Binary |

- **Total Records:** 768
- **Features:** 8 + 1 Target
- **Class Distribution:** ~65% Non-Diabetic | ~35% Diabetic *(slightly imbalanced but acceptable)*

---

## 📁 Project Structure

```
diabetes-eda/
│
├── 📓 diabetes_dataset.ipynb       # Main EDA notebook
├── 📄 diabetes.csv                 # Raw dataset
├── 📄 README.md                    # Project documentation (this file)
└── 📁 visuals/                     # Exported charts and plots (optional)
```

---

## 💡 Key Findings

### 🔑 Top Predictors of Diabetes

| Rank | Feature | Insight |
|---|---|---|
| 1 | **Glucose** | Strongest predictor — diabetic patients show significantly higher glucose levels |
| 2 | **BMI** | Higher BMI consistently linked with positive diabetes outcomes |
| 3 | **Age** | Diabetes prevalence increases steadily with age group |
| 4 | **Pregnancies** | Higher number of pregnancies correlates with diabetes risk |
| 5 | **Insulin** | High variance but moderate impact — heavily skewed distribution |

### 📌 Data Quality Issues Resolved

- **Zero values** in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were identified as physiologically impossible and treated as **missing values**.
- All missing values were **imputed using column medians** to preserve distribution integrity and avoid bias.

### 📌 Distribution Insights

- **Glucose** and **BMI** show strong variation across the diabetic vs non-diabetic groups.
- **Insulin** and **SkinThickness** exhibit **extreme outliers** — robust scaling is recommended for model training.
- **Clear separation** in the pairplot for `Glucose` and `BMI` when grouped by `Outcome`.

### 📌 Age Group Analysis

| Age Group | Diabetes Prevalence |
|---|---|
| 20s | Low |
| 30s | Moderate |
| 40s | High |
| 50s+ | Highest |

> Diabetes risk **increases significantly** after the age of 40.

---

## 🔄 EDA Workflow

```
Raw Data
   │
   ▼
📥 Data Loading & Inspection
   │ → df.shape, df.info(), df.describe()
   │
   ▼
🧹 Data Cleaning
   │ → Detect invalid zeros in medical features
   │ → Replace with NaN → Impute with median
   │
   ▼
📊 Univariate Analysis
   │ → Histograms for all numerical features
   │ → Countplot for target variable (Outcome)
   │
   ▼
📈 Bivariate Analysis
   │ → Boxplots: Glucose, BMI, Age vs Outcome
   │ → Barplots: Pregnancies, SkinThickness, Insulin vs Outcome
   │ → Violin Plots: Glucose, BMI, Age vs Outcome
   │
   ▼
🔗 Multivariate Analysis
   │ → Correlation Heatmap
   │ → Pairplot (hue = Outcome)
   │ → Joint Plot (Glucose vs BMI)
   │ → Age Group Segmentation
   │
   ▼
🤖 Predictive Modeling
   │ → Logistic Regression
   │ → Train/Test Split (80/20)
   │ → Accuracy Score + Classification Report
```

---

## 📉 Visualizations

The notebook includes the following visualizations:

- **Correlation Heatmap** — Feature interdependency matrix with `coolwarm` palette
- **Boxplots** — Distribution comparison of key features by Outcome
- **Violin Plots** — Density + distribution view of Age, Glucose, BMI by Outcome
- **Histograms** — Individual feature distributions across the entire dataset
- **Pairplot** — Pairwise scatterplots colored by diabetes outcome
- **Countplot** — Class distribution of Diabetic vs Non-Diabetic patients
- **Barplots** — Mean values of features across outcome groups
- **Jointplot** — KDE plot of Glucose vs BMI relationship
- **Age Group Countplot** — Diabetes prevalence per age decade

---

## 🤖 Predictive Modeling

A baseline **Logistic Regression** classifier was trained to demonstrate the predictive potential of the dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df.drop(['Outcome', 'Age_Group'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**Key Results:**
- The model achieves solid baseline accuracy using **Glucose, BMI, and Age** as the most influential features.
- Full classification report (precision, recall, F1-score) available in the notebook.

> 💡 This EDA forms the foundation for more advanced models like **Random Forest**, **XGBoost**, or **Neural Networks**.

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and cleaning |
| `numpy` | Numerical operations and NaN handling |
| `matplotlib` | Base plotting and figure configuration |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Train/test split, Logistic Regression, metrics |

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/diabetes-eda.git
cd diabetes-eda
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook diabetes_dataset.ipynb
```

Or open it directly in **Google Colab** by uploading the `.ipynb` file.

> ⚠️ Make sure `diabetes.csv` is in the same directory or update the file path inside the notebook.

---

## 🚀 Future Work

- [ ] Feature engineering (glucose-to-insulin ratio, BMI categories)
- [ ] Handle class imbalance using **SMOTE** or **class weighting**
- [ ] Try advanced classifiers: **Random Forest**, **XGBoost**, **SVM**
- [ ] Hyperparameter tuning with **GridSearchCV**
- [ ] Build an interactive dashboard with **Streamlit** or **Plotly Dash**
- [ ] Deploy a prediction API using **FastAPI** or **Flask**

---

## 👤 Author

**Mohammedasif Shethwala**
- 💼 LinkedIn:www.linkedin.com/in/asif-shethwala-96b632203
- 🐙 GitHub: https://github.com/easyon68-cloud

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

<div align="center">
  <strong>⭐ If you found this project useful, please give it a star!</strong>
</div>
