# Heart Disease Classification

This repository contains a machine learning project that predicts the likelihood of heart disease using the **BRFSS 2015 Heart Disease Health Indicators dataset**. The study explores multiple classification algorithms, feature engineering, and handling of class imbalance to improve healthcare predictions.
---

## 📌 Project Overview
Cardiovascular disease is the leading cause of death worldwide. Early detection and risk stratification can save lives and reduce the burden on healthcare systems. This project leverages **machine learning models** to identify individuals at risk of heart disease.
---

## ⚙️ Methodology
1. **Dataset**: BRFSS 2015 (253,680 records, 22 variables).
2. **Preprocessing**:
   - Addressed severe class imbalance (~9.4% positives) using undersampling.
   - Standardized numerical features (BMI, Age, MentHlth, PhysHlth).
   - Created second-order polynomial and interaction terms.
3. **Models Applied**:
   - Logistic Regression
   - Decision Tree
   - Naive Bayes
   - Linear Discriminant Analysis (LDA)
   - K-Nearest Neighbors (KNN)
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, AUC.

---

## 📊 Key Findings
- **Naive Bayes** achieved the best performance on the balanced dataset:
  - F1 Score = 0.985  
  - AUC = 0.999  
- **Logistic Regression with interaction terms** provided the most balanced performance after fine-tuning.  
- Class imbalance significantly impacted results, proving the importance of balancing strategies.  

---

## 🔑 Insights
- Handling **imbalanced datasets** is crucial for medical predictions.  
- **Feature engineering** (interaction & polynomial terms) improves model performance.  
- **Logistic Regression** remains interpretable and effective for clinical data, while **Naive Bayes** excels with high accuracy.  

---
