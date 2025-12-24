# Kaggle Assignment 3 – Sentiment Analysis

This repository contains my solution for **Kaggle Assignment 3** of the  
**Machine Learning Practice (MLP)** course, IIT Madras.

The objective of this assignment is to perform **multi-class sentiment analysis**
on movie review phrases using **classical machine learning models** and
**TF-IDF-based text features**, strictly within the MLP syllabus.

---

## Problem Statement
Given a movie review phrase, predict its **sentiment label (0–4)**.

---

## Approach
The complete workflow followed in this project includes:

- Loading and inspecting the dataset  
- Identifying text, target, and numeric features  
- Handling missing values and duplicates  
- Feature engineering (text length features)  
- Outlier detection and handling using IQR  
- Exploratory Data Analysis with visualizations  
- Text preprocessing using **TF-IDF**  
- Training **7+ machine learning models**  
- Hyperparameter tuning for **3 models**  
- Model comparison using validation metrics  
- Final model selection and Kaggle submission  

All steps are documented clearly inside the notebook.

---

## Models Used
- Logistic Regression  
- Linear SVC  
- Multinomial Naive Bayes  
- SGD Classifier  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Passive Aggressive Classifier  
- Character n-gram TF-IDF + Logistic Regression (final model)

---

> Note: Dataset files are not included in this repository and should be
downloaded directly from the Kaggle competition page.

---

## Running the Notebook

### Option 1: Google Colab
Open the notebook directly in Colab:


### Option 2: Kaggle
Upload the notebook to Kaggle and use the competition-provided datasets.

---

## Final Output
The notebook generates a `submission.csv` file in the required Kaggle format,
which can be uploaded to the competition to obtain the leaderboard score.

---

## Author
**Kashyap Rahul**  
IIT Madras – BS Degree Programme  
Machine Learning Practice



