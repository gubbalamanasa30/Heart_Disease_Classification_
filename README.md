# Heart-Disease-Classification

This project applies **machine learning** to predict the likelihood of heart disease using the **BRFSS 2015 Heart Disease Health Indicators dataset**. It explores multiple models, feature engineering techniques, and balancing strategies to improve predictive accuracy in healthcare.

---

## 📌 Project Overview
- Predict heart disease risk using health indicators.  
- Compare classification models (Logistic Regression, Decision Tree, Naive Bayes, LDA, KNN).  
- Apply data balancing, feature engineering, and model tuning.  
- Evaluate models with accuracy, precision, recall, F1-score, and AUC.  

---

## Background
Cardiovascular disease is the **leading cause of death worldwide**, responsible for ~17.9 million deaths annually (WHO). Early diagnosis and risk prediction can help save lives and reduce healthcare costs.  
The BRFSS 2015 dataset provides large-scale health survey data. Machine learning enables discovering **hidden patterns** and improving classification of at-risk individuals.

---

## Motivation
Heart disease risk detection is often hindered by **class imbalance**, where positive cases are underrepresented. This project addresses that gap through **undersampling, feature engineering, and tuning**, making predictions more reliable for healthcare applications.

---

## Objectives
- Preprocess and balance the dataset.  
- Build and train five machine learning classifiers.  
- Add second-order polynomial and interaction terms.  
- Tune models and compare their performance.  
- Identify the most reliable model for healthcare risk prediction.  

---

## Tech Stack
- **Programming Language**: R  
- **Techniques**: Data preprocessing, feature engineering, balancing (undersampling), model training, model tuning  
- **Libraries**: `caret`, `ggplot2`, `e1071`, `MASS`, `class`  

---

## 🔎 Methodology
1. **Data Preprocessing** – Handle imbalance with undersampling, standardize numerical features, create interaction and polynomial terms.  
2. **Exploratory Data Analysis (EDA)** – Summary statistics, correlation analysis, and visualizations (histograms, boxplots, bar plots).  
3. **Model Development** – Train Logistic Regression, Decision Tree, Naive Bayes, LDA, and KNN.  
4. **Model Tuning** – Apply stepwise selection (Logit), grid search (Tree, KNN, NB), cross-validation.  
5. **Evaluation** – Compare Accuracy, Precision, Recall, F1-score, AUC.  

---

## 📊 Key Findings
- **Naive Bayes** performed best on the balanced dataset:  
  - F1 Score = 0.985  
  - AUC = 0.999  
- **Logistic Regression with interaction terms** provided the most balanced performance after fine-tuning (F1 = 0.779, AUC = 0.849).  
- Class imbalance significantly reduced recall and F1 scores in Logistic Regression, Decision Tree, and LDA.  

---


---

## 👩‍💻 Author
Developed as a healthcare-focused machine learning project to demonstrate how **data preprocessing, feature engineering, and algorithm selection** impact predictive performance.

