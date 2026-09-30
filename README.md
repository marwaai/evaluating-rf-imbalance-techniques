# Glass Type Classification: Multi-Class Imbalance Strategy Analysis

This repository evaluates strategies for handling severe class imbalance in multi-class dataset classification using the **Glass Identification Dataset**[cite: 1]. The analysis compares algorithmic cost-sensitive learning (`class_weight='balanced'`) against data-level resampling (`RandomOverSampler`) using a **Random Forest Classifier**[cite: 1].

---

## Dataset Overview

* **Dataset**: Glass Identification Data (214 samples, 9 numerical features)[cite: 1].
* **Features**: Refractive Index (`RI`) and chemical compositions (`Na`, `Mg`, `Al`, `Si`, `K`, `Ca`, `Ba`, `Fe`)[cite: 1].
* **Target Column**: `Type` (6 glass types: `1`, `2`, `3`, `5`, `6`, `7`)[cite: 1].
* **Missing Values**: None[cite: 1].
* **Class Distribution**:
  * Type `2`: 76 instances[cite: 1]
  * Type `1`: 70 instances[cite: 1]
  * Type `7`: 29 instances[cite: 1]
  * Type `3`: 17 instances[cite: 1]
  * Type `5`: 13 instances[cite: 1]
  * Type `6`: 9 instances[cite: 1]

---

## Project Workflow

1. **Data Ingestion & Cleaning**: Loaded dataset, verified zero missing values across all features, and extracted target class counts[cite: 1].
2. **Train-Test Split**: Partitioned data into 80% training (`x_train`, `y_train`) and 20% test sets (`x_test`, `y_test`) with `random_state=42`[cite: 1].
3. **Model Strategy Comparison**:
   * **Model A**: `RandomForestClassifier(class_weight='balanced', random_state=42)`[cite: 1].
   * **Model B**: `RandomOverSampler(random_state=42)` paired with standard `RandomForestClassifier(random_state=42)`[cite: 1].
4. **Evaluation**: Evaluated performance on unseen test set using classification metrics (Precision, Recall, F1-Score)[cite: 1].

---

## Performance Summary

Both approaches achieved an overall accuracy of **86%** on the test set (`n=43`), but display distinct performance tradeoffs across minority classes[cite: 1]:

| Strategy | Overall Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Balanced Class Weights** | 86% | 0.92 | 0.83 | 0.85 | 0.85 |
| **Random Over-Sampling** | 86% | 0.88 | 0.92 | 0.90 | 0.86 |

### Classification Report Details

#### 1. Class Weighting (`class_weight='balanced'`)[cite: 1]
```text
              precision    recall  f1-score   support

           1       0.79      1.00      0.88        11
           2       0.85      0.79      0.81        14
           3       1.00      0.67      0.80         3
           5       1.00      0.50      0.67         4
           6       1.00      1.00      1.00         3
           7       0.89      1.00      0.94         8

    accuracy                           0.86        43
   macro avg       0.92      0.83      0.85        43
weighted avg       0.87      0.86      0.85        43
```[cite: 1]

#### 2. Random Over-Sampling (`RandomOverSampler`)[cite: 1]
```text
              precision    recall  f1-score   support

           1       0.75      0.82      0.78        11
           2       0.91      0.71      0.80        14
           3       0.75      1.00      0.86         3
           5       1.00      1.00      1.00         4
           6       1.00      1.00      1.00         3
           7       0.89      1.00      0.94         8

    accuracy                           0.86        43
   macro avg       0.88      0.92      0.90        43
weighted avg       0.87      0.86      0.86        43
```[cite: 1]

---

## Executive Strategy & Insights

* **Class Weighting (`class_weight='balanced'`)**: Preserves high precision across rare classes (1.00 precision on types 3, 5, and 6) while preventing potential overfitting caused by duplicating identical instances[cite: 1].
* **Random Oversampling (`RandomOverSampler`)**: Significantly improves recall for minority classes (improving recall on Type 3 from 0.67 to 1.00 and Type 5 from 0.50 to 1.00) at the cost of a slight drop in precision for majority/minority boundary cases[cite: 1].
* **Final Selection**: **Class Weighting** is prioritized for production setups where minimizing false positives on rare classes is critical, whereas **Oversampling** serves best when maximizing detection rate (recall) of rare glass types is the primary objective[cite: 1].
