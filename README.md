# Credit Risk Prediction

This project develops and compares multiple machine learning models to classify loan defaults using the LendingClub dataset. It explores data preprocessing, feature engineering, class imbalance handling, and model evaluation to build a robust credit risk prediction pipeline. This project is ongoing.

## Project Overview
The goal is to predict whether a loan applicant will default or repay based on historical application and credit data. We evaluate and compare three models:
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

## Features
- Data Preprocessing:
  - Missing value handling
  - Categorical encoding (OneHotEncoder)
  - Feature scaling (StandardScaler)
- Class Imbalance Handling:
  - SMOTE oversampling
  - Class weighting
  - Threshold optimization for better recall
- Model Training and Tuning:
  - Hyperparameter tuning via GridSearchCV and RandomizedSearchCV
  - ROC AUC maximization
- Model Evaluation:
  - Confusion Matrix
  - Classification Report
  - ROC Curve and AUC score

## Project Structure
'''
credit-risk-prediction/
│
├── data/                  # (ignored in Git) Large datasets stored locally
├── notebooks/             # Jupyter notebooks for each model
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_xgboost.ipynb
│
├── .gitignore             # Ensures large data files are not pushed to GitHub
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
'''

## Results Summary
| Model                | ROC AUC | Accuracy | Notes |
|----------------------|---------|----------|-------|
| Logistic Regression  | 0.716   | 75%      | High recall for defaulters but high false positive rate |
| Random Forest        | 0.716   | 75%      | Balanced precision-recall, robust performance |
| XGBoost              | 0.724   | 69%      | Improved minority recall, optimized threshold |

## Installation and Usage
### 1. Clone the Repository
git clone https://github.com/ZeideKayyal/credit-risk-prediction.git  
cd credit-risk-prediction

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Add Data Files Locally
This repo does not include the LendingClub dataset due to size limits. Create a `data/` directory and place the required CSV files there:
data/
├── accepted_2007_to_2018Q4.csv
├── rejected_2007_to_2018Q4.csv
└── cleaned_credit_data.csv

You can download the raw datasets from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

### 4. Run the Notebooks
Open Jupyter Notebook or Jupyter Lab:
jupyter notebook

Then execute:
- 01_logistic_regression.ipynb
- 02_random_forest.ipynb
- 03_xgboost.ipynb

## Skills and Tools Used
Languages and Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib  
Techniques: Data Preprocessing, Feature Engineering, SMOTE, Class Weighting, Hyperparameter Tuning, Model Evaluation  
Tools: Git/GitHub, Jupyter Notebook

## Next Steps
- Add LightGBM for comparison
- Implement feature importance analysis
- Explore deep learning approaches (e.g., TabNet, MLP)
- Deploy as a web app for credit risk scoring
