# **Task 01 - Term Deposit Subscription Prediction (Bank Marketing)**

## **Problem Statement**

When running marketing campaigns, banks invest significant time and resources into contacting customers. Without a targeted approach, campaigns can be inefficient, leading to wasted effort on customers who are unlikely to convert. 

This task focuses on developing a predictive pipeline using a real-world dataset from the UCI Machine Learning Repository. The dataset contains demographic and campaign interaction data for bank customers. The goal is to predict whether a customer will subscribe to a term deposit.

The core challenge is to answer the following questions and build a robust pipeline:
- How do we handle and encode categorical text data so a machine learning model can process it?
- Can we build a classification model (like Random Forest) to accurately predict customer subscriptions?
- How do we evaluate the model's true performance, especially when dealing with imbalanced classes (where most people say "No")?
- How can we use Explainable AI (XAI) to look inside the "black box" and understand exactly *why* the model makes a specific prediction?

---

## **Approach**

### ***1. Environment Setup***
Imported the required libraries: `pandas` for data handling, `matplotlib` and `seaborn` for visualization, `sklearn` for preprocessing, modeling, and evaluation, and `shap` for model interpretability. 

### ***2. Data Loading and Inspection***
- Loaded the bank marketing dataset (`bank-full.csv`) using `pandas` with the correct separator (`sep=';'`).
- Inspected the dataset structure using `.head()` to view the first 5 rows and understand the features (age, job, marital, balance, etc.).
- Checked for missing values using `.isnull().sum()`. The dataset was confirmed to be completely clean with zero missing values.

### ***3. Feature Encoding***
- Machine learning models require numerical input. Text-based categorical variables were converted using two techniques:
  - **Label Encoding:** Applied to the target variable (`y`) to convert "yes" and "no" into `1` and `0`.
  - **One-Hot Encoding:** Applied to the remaining categorical features using `pd.get_dummies()`. The `drop_first=True` argument was used to avoid the dummy variable trap, creating distinct binary columns for each category.

### ***4. Model Training and Evaluation***
- **Training:** Split the dataset into training (80%) and testing (20%) sets. Initialized and trained a `RandomForestClassifier` with 100 decision trees (`n_estimators=100`).
- **Confusion Matrix:** Generated a matrix to view the exact breakdown of True Positives, True Negatives, False Positives, and False Negatives.
- **Metrics Analysis:** Checked the overall accuracy (~90.4%). Calculated the F1-Score (~0.51) to get a balanced view of Precision and Recall, which is crucial for imbalanced datasets.
- **ROC Curve:** Plotted the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to visualize the model's ability to distinguish between subscribers and non-subscribers.

### ***5. Model Interpretability (SHAP)***
- **Explainer Initialization:** Set up a `shap.TreeExplainer` specifically designed to interpret the Random Forest model.
- **Summary Plot:** Generated a global summary plot to identify which features (e.g., call duration, age, balance) had the most significant overall impact on the model's decision-making process.
- **Waterfall Plots:** Extracted the first 5 test predictions and generated individual Waterfall plots. This provided a local, step-by-step breakdown of how each specific feature contributed to a single customer's final prediction.

---

## **Key Findings**

- The dataset is clean and requires no missing value imputation.
- The Random Forest model achieved a high overall accuracy of ~90.4%, but the F1-score of ~0.51 indicates a **class imbalance**. The model is highly accurate at identifying customers who will *not* subscribe but struggles slightly to capture all the actual subscribers.
- The Confusion Matrix confirms this, showing a higher rate of False Negatives compared to False Positives. 
- Using SHAP (Explainable AI) successfully demystifies the model's behavior, allowing us to pinpoint the exact customer traits and campaign metrics that drive a "Yes" prediction. 
- These insights provide actionable intelligence for the bank, allowing them to shift from blind cold-calling to highly targeted, data-driven marketing campaigns.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, inspection, and manipulation |
| `matplotlib.pyplot` | Base plotting framework for the ROC Curve |
| `seaborn` | Advanced statistical visualizations |
| `sklearn.preprocessing` | Feature encoding (`LabelEncoder`) |
| `sklearn.ensemble` | Model building (`RandomForestClassifier`) |
| `sklearn.metrics` | Model evaluation (`accuracy_score`, `confusion_matrix`, `f1_score`, `roc_curve`) |
| `shap` | Explainable AI (XAI) for global and local model interpretability |