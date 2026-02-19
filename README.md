🎓 Student Score Prediction using Machine Learning
📌 Project Overview

This project predicts students’ exam scores based on academic and lifestyle factors using regression techniques. The objective is to analyze how study behavior and related features influence academic performance and to compare linear and non-linear regression models.

📊 Dataset

Student Performance Factors Dataset (Kaggle)
The dataset includes numerical and categorical features such as:
Hours studied
Attendance
Sleep duration
Previous academic scores
Motivation and lifestyle indicators

🔍 Exploratory Data Analysis (EDA)
Statistical summary of numerical features
Distribution analysis of exam scores
Relationship analysis between study hours and exam performance
Identification of missing values and data types

🛠️ Methodology
Data Preprocessing
Feature selection
Handling missing values
Encoding categorical variables
Train–test split (80/20)
Regression Models
Linear Regression
Polynomial Regression (degree = 2)

Algorithm Implementation
In addition to using scikit-learn’s LinearRegression, the linear regression algorithm was also implemented from scratch using NumPy to understand the underlying mathematics.
The custom implementation was evaluated using the same train/test split
Evaluation metrics closely matched scikit-learn’s implementation
The scikit-learn model is used by default for robustness and reproducibility

📈 Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

✅ Results
The linear regression model achieved an R² score of 0.64, indicating a strong relationship between study-related features and exam performance. Polynomial regression provided marginal improvement, highlighting mild non-linearity in the data.

🧰 Tech Stack
Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

📁 Project Structure
.
├── data/
│   └── student_data.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── linearregression_from_scratch.py
├── run.py
├── requirements.txt
└── README.md


🚀 How to Run
pip install -r requirements.txt
python run.py