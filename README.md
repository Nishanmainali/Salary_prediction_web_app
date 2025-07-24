# Salary Prediction Using Machine Learning

## Table of Contents
- [Overview](#overview)
- [Demo](demo)
- [Pipeline structure](#pipeline-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Results & Insights](#results--insights)
- [Future Work](#future-work)


---

## Overview

This data science project predicts the annual salary of tech professionals based on their demographics and skills. It uses features such as programming languages, platforms, country, education level, and years of experience. The project involves data preprocessing, feature engineering (multi-label one-hot encoding), model training using XGBoost, and deployment of an interactive web app using Streamlit.

---

## Demo

<img width="979" height="1047" alt="Screenshot 2025-07-24 at 12 45 31 PM" src="https://github.com/user-attachments/assets/ec4bf325-d805-481b-9689-6c8808950a84" />


---


## Pipeline Structure
<img width="736" height="267" alt="model_pipeline" src="https://github.com/user-attachments/assets/52300c94-b645-4c84-9429-c1fa38b76590" />



---
## Dataset

- **Source:** Stack Overflow Developer Survey 2023
- **Size:** ~33000 responses (after cleaning)
- **Target Variable:** `Salary` (Annual Salary)
- **Key Features Used:**
  - `Country`
  - `EdLevel` (Education level)
  - WorkMode( Hybrid' 'Remote' 'In-person') 
  - OrgSize  (Organization Size)
    - Small(2 to 99)   
    - Medium(100 to 499)
    - Large(1,000 to 4,999)
    - Enterprise(More than 4,999)
    - Freelancer
    - I don’t know
  - `DevType` (Grouped into broader job roles)
  - `Experience` (Years of professional coding experience)
  - `Language` (Multi-label field)
  - 'DataBase' (Multi_label field)
  - `Platform` (Multi-label field)
  

---

## Features

- **Multi-label One-Hot Encoding:** Custom transformer used for 'Language', 'DataBase', and 'Platform'.
- **Label Grouping:** Grouped `DevType` into simplified roles like Full-Stack Developer, Data Scientist, etc.
- **Standardization:** Applied on numerical fields like experience
- **Machine Learning Model:** XGBoost Regressor
- **Web App Deployment:** Streamlit-based interface for user interaction

---

## Technologies Used

- **Language:** Python 3.12
- **Libraries & Tools:**
  - Data Handling: `pandas`, `numpy`
  - ML & Modeling: `scikit-learn`, `xgboost`
  - Web App: `streamlit`
  - Visualization: `matplotlib`, `seaborn`
- **Custom Code:** `MultiLabelBinarizerDF` (Sklearn-compatible transformer for multi-label one-hot encoding)

---

## Project Structure
```text
salary_prediction/
├── survey_results_public.csv           # Raw data
├── model_pipeline.ipynb                # Pipeline for the model              
├── custom_transformers.py              # Custom transformer class
├── salary_app.py                       # Streamlit app
├── model_pipeline.pkl                  # Pickled ML pipeline
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```
---

## Model Performance

| Metric               | Value          |
|----------------------|----------------|
| R² Score (Test Set)  | ~0.64          |
| Mean Absolute Error  | ~$23,000        |
| Model Used           | XGBoostRegressor |

---

## Results & Insights

-  Developers who know **Python**, **SQL**, or **JavaScript** tend to have higher salaries.
-  Country is one of the strongest predictors of salary.
-  While higher education may offer a slight edge, it’s not a strong predictor of salary
   in tech-related fields—real-world skills and roles seem to play a bigger role.
-  Executive/Manager and DevOps roles have the highest median salaries, while Mobile Developers earn the least, with all roles
   showing significant salary variation and outliers.

--- 

## Future Work

- **Improve Model Accuracy with Deep Learning:** Future iterations can explore advanced models such as deep neural networks 
  (DNNs) or transformers, which may capture more complex nonlinear relationships in the data and further improve predictive performance.
- **Globalization Support:** Expand the model to better support non-English responses and country-specific salary standards
  to improve inclusivity and accuracy across regions.

















