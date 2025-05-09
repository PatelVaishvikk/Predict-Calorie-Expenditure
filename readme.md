# 🧠 Predicting Calorie Expenditure - Kaggle Playground Series S5E5

This repository contains my complete solution for the Kaggle competition: **"Playground Series - Season 5, Episode 5"**, where the task is to **predict calories burned** during physical activity based on physiological and activity-related features.

## 🚀 Project Overview

- **Goal**: Predict the number of calories burned (`Calories`) using input features such as Age, Sex, Height, Weight, Duration, Heart Rate, and Body Temperature.
- **Dataset Size**: 750,000 training records and 75,000 test records.
- **Evaluation Metric**: RMSLE (Root Mean Squared Logarithmic Error)

---

## 📁 Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with known calorie outputs |
| `test.csv` | Test data without target variable |
| `sample_submission.csv` | Sample format for submission |
| `submission.csv` | My model's final predictions |
| `Predict_Calorie_Expenditure.ipynb` | Complete notebook with EDA, feature engineering, and model training |
| `readme.md` | You're reading it |

---

## 📊 Exploratory Data Analysis (EDA)

1. **Shape & Dtypes**: No missing values; all features well-typed.
2. **Target Distribution**: Skewed, so we applied `log1p()` transform.
3. **Correlation Analysis**: Duration, Heart Rate, and Body Temp are most correlated with Calories.
4. **Sex-based Differences**: Visualized calorie burn differences between male and female participants.
5. **Histograms**: To inspect distribution of numeric features.

---

## 🛠️ Feature Engineering

- **Sex Encoding**: Converted `Sex` from string to numeric (`male=1`, `female=0`)
- **BMI**: Added `BMI = Weight / (Height/100)^2`
- **Duration_HeartRate**: Created interaction feature `Duration * Heart Rate`

These new features improved model performance by introducing physiological logic.

---

## 🔍 Model Building & Evaluation

### 🔹 Model 1: Linear Regression

- Used log-transformed target.
- Validation RMSLE: **0.179**

### 🔹 Model 2: CatBoost Regressor

- Automatically handles categorical features.
- Robust to skewed features and missing values.
- Validation RMSLE: **0.059**

### 🔬 Evaluation Metric

We used **RMSLE** to reduce the penalty for small differences and stabilize variance due to target skewness.

```python
from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

📈 Final Prediction & Submission
Final model used: CatBoostRegressor

Predictions were inverse transformed using expm1().

Saved in submission.csv.

sample['Calories'] = np.expm1(cat_model.predict(X_test))
sample.to_csv('submission.csv', index=False)
🧠 Key Learnings
Feature transformations and engineered features like BMI and Duration×HeartRate significantly improved results.

CatBoost performed much better than Linear Regression due to its ability to capture nonlinear interactions.

Log-transforming the target helped stabilize variance and improved generalization.

🏁 Next Steps
Tune CatBoost hyperparameters using GridSearchCV

Try more advanced models like XGBoost or LightGBM

Apply PCA or Lasso for feature selection

Build ensemble models

📌 Author
Vaishvik Patel

Data Analyst | ML Enthusiast | Kaggle Competitor

LinkedIn | GitHub

