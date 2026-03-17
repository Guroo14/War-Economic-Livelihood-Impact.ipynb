# War-Economic-Livelihood-Impact.ipynb
A machine learning project that predicts war profiteering occurrence based on economic indicators during armed conflicts using XGBoost classification.
# War Economic Impact — Profiteering Prediction

A machine learning project that predicts war profiteering occurrence based on
economic indicators during armed conflicts using XGBoost classification.

##  Project Overview

This project analyzes the economic impact of wars and predicts whether
war profiteering is documented, based on features such as GDP change,
inflation rate, unemployment, and black market activity.

Target Variable: `War_Profiteering_Documented` (Binary: 0 = No, 1 = Yes)

Data Preprocessing

Dropped columns:
- `Unemployment_Spike_Percentage_Points` — derived column, correlation > 0.97
- `Primary_Black_Market_Goods` — 156 unique categories, high cardinality noise
- `Conflict_Name`, `Primary_Country` — ID columns, no predictive value

Encoding
Nominal columns (`Region`, `Conflict_Type`) → LabelEncoder
Binary columns (`Status`, `War_Profiteering_Documented`) → Manual mapping

**Feature Scaling:** StandardScaler (required for PCA)

**PCA:** Reduced dimensions to capture 95% of variance


Handled using `scale_pos_weight` parameter in XGBoost.

## Results

 Metric 

 ROC-AUC 
 F1-Score (weighted) 
 F1-Score (macro) 
 Precision (class 1)
 Recall (class 1)

Primary metric: ROC-AUC** — chosen due to class imbalance.
Secondary metric: Recall** — minimizing false negatives is critical
 as missing actual profiteering cases is costlier than false alarms.

 Visualizations

- Feature Importance (XGBoost)
- Prediction Probability Distribution
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Correlation Heatmap
- PCA Scree Plot & Cumulative Variance
