# **Task 02 - Credit Risk Prediction**

## **Problem Statement**

In the financial industry, accurately assessing whether a loan applicant will default is critical for minimizing risk and making sound lending decisions. Without a systematic, data-driven approach, this process is inconsistent and prone to human bias.

This task focuses on building a complete machine learning pipeline — from raw data ingestion and cleaning through to model training and evaluation — using the **Loan Prediction Dataset** from Kaggle. The dataset contains applicant demographic, financial, and loan details across 614 records and 13 columns.

The core challenge is to answer the following questions using data preprocessing, visualization, and classification modeling:

- How should missing values across multiple feature types be appropriately handled?
- What are the distributions and key relationships among features like loan amount, income, education, and credit history?
- Which features are most predictive of loan approval or rejection?
- How well can Logistic Regression and Random Forest models classify loan outcomes?
- How should model performance be interpreted using accuracy, confusion matrix, and classification report?

---

## **Approach**

### ***1. Environment Setup***

Imported the required libraries: `pandas` and `numpy` for data manipulation, `matplotlib` and `seaborn` for visualization, and `sklearn` for preprocessing, modeling, and evaluation. The dataset was retrieved programmatically from Kaggle using the `kagglehub` library.

### ***2. Data Loading and Inspection***

- Loaded the dataset using `pd.read_csv()` after resolving the Kaggle download path.
- Inspected structure using `.head()`, `.tail()`, `.shape`, `.columns`, `.info()`, and `.describe()`.
- The non-informative `Loan_ID` column was dropped immediately.
- Missing value check via `.isnull().sum()` revealed nulls across seven columns: `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.

### ***3. Handling Missing Values***

A feature-aware imputation strategy was applied rather than a blanket fill:

- **Gender, Dependents, Married, Credit_History**: Filled with mode to preserve the dominant category.
- **Self_Employed**: Filled using group-wise mode per `Education` level, capturing the relationship between employment type and education.
- **LoanAmount / Loan_Amount_Term**: Filled with the column mean, appropriate for continuous numerical features.
- A new engineered feature `Total_Income` was created by summing `ApplicantIncome` and `CoapplicantIncome` to represent total household earning capacity.
- Post-imputation verification confirmed zero null values across all columns.

### ***4. Visualizations***

#### Histograms

- `Education` and `Self_Employed` showed clear bimodal distributions reflecting their binary categorical nature.
- `LoanAmount` was right-skewed, indicating most applicants requested moderate amounts with a long tail of high-value requests.
- `Total_Income` was also right-skewed with visible outliers, reflecting high income variability among applicants.
- `Credit_History` displayed a bimodal distribution — the large majority had a positive credit history (1.0), with a small proportion having none (0.0).

#### Box Plots

- `LoanAmount` and `Total_Income` box plots confirmed numerous upper outliers, validating the need for robust transformation before modeling.
- `Credit_History` showed zeros as apparent outliers due to the binary nature of the variable — a distributional characteristic, not a data quality issue.

#### Scatter Plots & Pair Plot

- A scatter plot of `Total_Income` vs. `LoanAmount`, color-coded by `Credit_History`, showed applicants with positive credit history distributed broadly across income and loan ranges.
- A pair plot of the three key numerical features confirmed weak linear correlations between them.
- A **Pearson correlation heatmap** quantified low inter-feature correlation, with `Credit_History` being the most independent predictor.

#### Skewness Analysis

- Computed skewness values confirmed that `Total_Income` and `LoanAmount` were substantially right-skewed, directly motivating the log transformation applied during preprocessing.

### ***5. Data Preprocessing***

- Applied `np.log()` to `LoanAmount` and `Total_Income` to reduce skewness and compress extreme values.
- Applied `RobustScaler` (using median and IQR) to the log-transformed features — preferred over `StandardScaler` given the persistent extreme income values.
- Binary columns (`Gender`, `Married`, `Education`, `Self_Employed`) were encoded using a custom `map()` dictionary for interpretability.
- `Dependents` `"3+"` string values were converted to integer `3` before casting the column to `int`.
- `Property_Area` (Urban, Semiurban, Rural) was one-hot encoded using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.
- Target variable `Loan_Status` was encoded as `1` (Approved) and `0` (Rejected).

### ***6. Model Training***

The dataset was split using an **80/20 train-test split** with `random_state=42` for reproducibility. Two classification models were trained:

#### Logistic Regression

- Trained as an interpretable baseline model on 10 selected features.
- Feature importance was visualized using absolute coefficient values — `Credit_History` and `Total_Income_log` emerged as the strongest predictors.

#### Random Forest Classifier

- Trained with 100 estimators and `class_weight='balanced'` to handle class imbalance between approved and rejected loans.
- Feature importances from the ensemble were visualized via a horizontal bar chart, confirming `Credit_History` and `Total_Income_log` as top contributors.

### ***7. Model Evaluation***

- **Accuracy Score**: Computed for both models to provide a high-level performance measure.
- **Classification Report**: Precision, recall, and F1-score evaluated per class (Approved/Rejected) to assess model behavior beyond overall accuracy.
- **Confusion Matrix Heatmap**: Visualized using `sns.heatmap()` with annotated counts, revealing the distribution of True Positives, True Negatives, False Positives, and False Negatives — providing actionable insight into each model's error types.

---

## **Key Findings**

- The dataset had missing values across 7 columns, all resolved using feature-appropriate imputation strategies.
- `LoanAmount` and `Total_Income` were heavily right-skewed with significant outliers; log transformation followed by robust scaling effectively normalized these features for modeling.
- **Credit_History** was the single most predictive feature — applicants with a positive credit history were overwhelmingly more likely to receive loan approval.
- **Total_Income** (combined applicant and co-applicant income) ranked as the second most important predictor, reflecting that household earning capacity is a strong signal of creditworthiness.
- Logistic Regression provided an interpretable and competitive baseline, while Random Forest with balanced class weights improved recall on the minority class (rejected loans).
- Confusion matrix analysis highlighted that false positives (rejected applicants classified as approved) represent the costlier error type in a real lending context, guiding model selection toward higher precision on approvals.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, inspection, cleaning, and feature engineering |
| `numpy` | Numerical operations and log transformation |
| `matplotlib` | Base plotting for histograms, box plots, and scatter plots |
| `seaborn` | Statistical visualizations including heatmaps and pair plots |
| `sklearn.preprocessing` | RobustScaler, LabelEncoder, OneHotEncoder |
| `sklearn.model_selection` | train_test_split for 80/20 data partitioning |
| `sklearn.linear_model` | Logistic Regression for binary classification baseline |
| `sklearn.ensemble` | Random Forest Classifier for ensemble-based prediction |
| `sklearn.metrics` | accuracy_score, confusion_matrix, classification_report |
| `kagglehub` | Programmatic dataset download from Kaggle |
