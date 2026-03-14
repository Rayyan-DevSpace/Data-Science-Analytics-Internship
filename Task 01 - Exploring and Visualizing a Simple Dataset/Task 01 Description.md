# **Task 01 - Exploring and Visualizing a Simple Dataset**

## **Problem Statement**

When working with any dataset for the first time, it is essential to understand its structure, content, and statistical properties before applying any analytical or machine learning techniques. Without this foundational step, patterns can be missed, assumptions can be violated, and models can be built on poorly understood data.

This task focuses on developing the ability to load a real-world dataset, inspect it programmatically, and communicate its characteristics through visualizations. The **Iris dataset** is used as the subject of exploration. It is a classic benchmark in data science containing measurements of 150 iris flowers across three species.

The core challenge is to answer the following questions using only data inspection and visualization:

- What does the dataset look like structurally (shape, columns, sample rows)?
- Are there any missing values or data quality issues?
- What are the statistical properties of each feature?
- How are individual features distributed across the dataset?
- Are there observable relationships or separability between features across species classes?
- Which features are most useful for distinguishing between the three iris species?

---

## **Approach**

### ***1. Environment Setup***

Imported the required libraries: `pandas` and `numpy` for data handling, and `matplotlib` and `seaborn` for visualization. Also explored the datasets available in `sklearn` to understand the broader ecosystem before selecting the Iris dataset.

### ***2. Data Loading and Inspection***

- Loaded the Iris dataset using `sklearn.datasets.load_iris()` and converted it into a pandas `DataFrame` with proper column names (`sepal length`, `sepal width`, `petal length`, `petal width`).
- Added the `target` column (species label: 0 = Setosa, 1 = Versicolor, 2 = Virginica) to the DataFrame.
- Inspected the dataset structure using `.head()`, `.shape`, and `.columns`.
- Checked for missing values using `.isnull().sum()` - the dataset was confirmed to be clean with no nulls.

### ***3. Statistical Analysis***

- Used `.describe()` to extract summary statistics (mean, standard deviation, min, max, quartiles) for all four numerical features.
- This provided a quantitative baseline for understanding the scale and spread of the data before visualizing it.

### ***4. Visualizations***

#### Histograms

- Plotted individual histograms for each feature to examine the distribution of values.
- Petal length and petal width showed bimodal-like distributions, hinting at class separability, while sepal width appeared roughly normally distributed.

#### Box Plots

- Created box plots per feature to understand the spread, median, and presence of outliers across the full dataset.
- Sepal width showed the most outliers; petal dimensions had tighter interquartile ranges within classes.

#### Scatter Plots

- Plotted pairwise scatter plots between features, color-coded by species class, to visually assess linear separability.
- Key finding: petal length vs. petal width provided the clearest separation between Setosa and the other two species.

#### Pair Plot

- Generated a full pair plot using `sns.pairplot()` with `hue="target"` to get a comprehensive view of all feature-pair relationships in a single figure.
- Confirmed that **Setosa (class 0)** is clearly linearly separable from Versicolor and Virginica, while **Versicolor (class 1)** and **Virginica (class 2)** show some overlap, particularly in sepal measurements.

---

## **Key Findings**

- The Iris dataset is clean with no missing values across all 150 records and 4 features.
- Petal length and petal width are the most discriminative features for separating species.
- Setosa is distinctly separable from the other two classes across nearly all feature combinations.
- Versicolor and Virginica overlap in sepal space but are more separable in petal space.
- These insights lay the groundwork for applying classification models in subsequent tasks.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, inspection, and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting framework |
| `seaborn` | Statistical visualizations (box plots, pair plots) |
| `sklearn.datasets` | Loading the built-in Iris dataset |