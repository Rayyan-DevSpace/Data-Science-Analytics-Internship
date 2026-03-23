# **Task 03 - Predicting Insurance Claim Amounts**

## **Problem Statement**

Estimating medical insurance charges is a key challenge for insurers, healthcare providers, and policymakers alike. The ability to predict how much an individual is likely to claim — based on personal attributes such as age, BMI, smoking status, and region — enables better risk assessment, pricing strategies, and resource planning.

This task focuses on building a complete regression pipeline using the **Medical Cost Personal Dataset** from Kaggle. The dataset contains 1338 records with 7 features describing individual health and demographic characteristics, with `charges` as the continuous target variable.

The core challenge is to answer the following questions using data exploration, visualization, and regression modeling:

- What are the distributions and skewness characteristics of the key numerical features?
- How do age, BMI, and smoking status individually and collectively impact insurance charges?
- Which features carry the most predictive weight in a Linear Regression model?
- How accurately can the model estimate insurance charges, as measured by MAE, RMSE, and R²?
- Can the continuous predictions be meaningfully interpreted through a classification lens (high-cost vs. low-cost)?

---

## **Approach**

### ***1. Environment Setup***

Imported the required libraries: `pandas` and `numpy` for data handling, `matplotlib` and `seaborn` for visualization, and `sklearn` for preprocessing, model training, and evaluation. The dataset was downloaded programmatically from Kaggle using the `kagglehub` library and loaded directly via `pd.read_csv()`.

### ***2. Data Loading and Inspection***

- Loaded `insurance.csv` and previewed the structure using `.head()` and `.describe()`.
- Confirmed dataset dimensions: **1338 rows × 7 columns** (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
- Verified data quality using `.isnull().sum()` — the dataset was confirmed to be **fully clean with no missing values**.
- Used `.info()` to confirm data types: numerical columns (`age`, `bmi`, `children`, `charges`) and categorical columns (`sex`, `smoker`, `region`).

### ***3. Visualizations***

#### Histograms & Skewness

- **Age**: Histogram revealed a near-uniform distribution across age groups with slight right skewness — indicating fairly balanced representation of all adult age ranges.
- **BMI**: Displayed an approximately normal distribution with minimal skewness, centered around the overweight range (~30).
- **Children**: Right-skewed distribution — the majority of policyholders have 0–2 dependents, with very few having 4 or 5.
- **Smoker**: Bimodal binary distribution showing a clear majority of non-smokers over smokers.
- **Region**: Roughly uniform distribution across four U.S. geographic regions (northeast, northwest, southeast, southwest).
- **Charges**: Strongly right-skewed distribution with a long upper tail, indicating that most individuals incur moderate costs while a smaller group drives extremely high claims.

Quantitative skewness values were computed for `age`, `bmi`, `children`, and `charges` to numerically validate these observations and guide preprocessing decisions.

#### KDE Plots

- Side-by-side KDE plots were generated for `children`, `bmi`, and `charges` — both before and after scaling — to visualize the shape of each distribution and observe the normalization effect of `RobustScaler`.
- Post-scaling KDE plots confirmed a tighter, more centered distribution for `bmi` and `charges`, with outlier influence reduced.

#### Box Plots

- Box plots for `children` and `charges` confirmed the presence of upper outliers in both columns.
- `charges` showed a pronounced number of high-value outliers, reinforcing the right-skewed nature of the target variable.

#### Key Feature Visualizations (Age, BMI, Smoker vs. Charges)

- **Age vs. Charges** (scatter, hue = smoker): Revealed a clear upward trend — older individuals tend to incur higher charges. Smokers clustered distinctly at a higher charge band across all age groups.
- **BMI vs. Charges** (scatter, hue = smoker): Showed that high BMI alone does not linearly drive charges, but the combination of high BMI and smoking status created a visibly distinct high-cost cluster.
- **Smoker vs. Charges** (box plot): Provided the most striking visual — smokers had a dramatically higher median and wider range of charges compared to non-smokers, confirming smoking as the dominant cost driver.

#### Correlation Heatmap

- A Pearson correlation matrix was computed on the four numerical features (`age`, `bmi`, `children`, `charges`).
- `age` showed the strongest positive correlation with `charges` among the numerical features.
- `bmi` had a moderate positive correlation with `charges`.
- `children` showed near-zero correlation with `charges`, indicating limited predictive contribution on its own.

### ***4. Data Preprocessing***

- **Encoding**: Categorical columns `sex`, `smoker`, and `region` were one-hot encoded using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity. This expanded the feature set to include binary indicators such as `smoker_yes` and `region_northwest`, etc.
- **Feature/Target Split**: The dataset was vertically split into `X` (all features except `charges`) and `y` (`charges`).
- **Train-Test Split**: An **80/20 horizontal split** was applied using `train_test_split()` with `random_state=42`.
- **Scaling**: `StandardScaler` was applied to `X_train` and `X_test` (fit on train only, transformed on both) to normalize feature magnitudes before passing to the linear model.

### ***5. Model Training***

- A **Linear Regression** model was trained on the scaled training data using `sklearn.linear_model.LinearRegression`.
- Predictions were generated on the scaled test set using `lr.predict(X_test_scaled)`.
- Model coefficients (`lr.coef_`) were extracted for feature importance analysis.

#### Feature Importance

- Absolute coefficient values were computed and sorted to rank features by their influence on predicted charges.
- A horizontal bar chart visualized the ranking — **`smoker_yes`** had by far the largest absolute coefficient, confirming smoking status as the most impactful predictor.
- `age` and `bmi` ranked next, aligning with the visual findings from EDA.
- `children` and region-based features contributed minimally to the model's predictions.

### ***6. Model Evaluation***

#### Regression Metrics

- **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual charges — directly interpretable in dollar terms.
- **RMSE (Root Mean Squared Error)**: Penalizes large prediction errors more heavily than MAE, making it sensitive to high-cost outliers present in the dataset.
- **R² Score**: Indicates the proportion of variance in charges explained by the model — providing a normalized measure of overall fit quality.

#### Classification-Style Evaluation

- As a supplementary analysis, predictions were binarized using the **median charge** as a threshold — values above the median were labeled as high-cost (1), and below as low-cost (0).
- A **classification report** (precision, recall, F1-score) and **confusion matrix heatmap** were generated on these binarized labels to assess how well the regression model separates high-cost from low-cost cases.
- This dual evaluation approach bridges regression output with classification interpretability, offering a practical view of the model's decision-making quality in a real-world insurance context.

---

## **Key Findings**

- The dataset was fully clean with no missing values, requiring no imputation — only encoding and scaling were needed before modeling.
- **Smoking status** was overwhelmingly the most influential feature — smokers incurred dramatically higher charges across all age and BMI ranges, confirmed through both EDA visualizations and model coefficients.
- **Age** was the strongest continuous predictor of charges, with a consistent upward trend visible in scatter plots and reflected in a high absolute coefficient.
- **BMI** showed moderate predictive importance, but its impact was primarily amplified in combination with smoking status rather than acting as a standalone predictor.
- **Children** and **region** had minimal individual impact on charges, though they were retained as features for completeness.
- The right-skewed distribution of `charges` and presence of high-value outliers meant that RMSE was notably higher than MAE — indicating that the model struggles more with extreme high-cost cases than with typical claims.
- The binarized classification evaluation confirmed the model can effectively separate high-cost from low-cost cases, validating its practical utility beyond just raw charge estimation.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, inspection, and feature engineering |
| `numpy` | Numerical operations and skewness computation |
| `matplotlib` | Base plotting for histograms, box plots, and bar charts |
| `seaborn` | Statistical visualizations including scatter plots, box plots, and heatmaps |
| `sklearn.preprocessing` | StandardScaler and RobustScaler for feature normalization |
| `sklearn.model_selection` | train_test_split for 80/20 data partitioning |
| `sklearn.linear_model` | Linear Regression for continuous charge prediction |
| `sklearn.metrics` | MAE, RMSE, R², classification_report, confusion_matrix |
| `kagglehub` | Programmatic dataset download from Kaggle |
