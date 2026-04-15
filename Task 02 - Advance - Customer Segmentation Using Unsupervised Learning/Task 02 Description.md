# **Task 02 - Customer Segmentation Using Unsupervised Learning**

## **Problem Statement**

In the retail world, "one size fits all" marketing doesn't work. To maximize profit, a business needs to understand different types of customers and their spending habits. Since the data doesn't come with pre-defined labels (like "Good Customer" or "Bad Customer"), we use **Unsupervised Learning** to discover hidden groups.

This task explores the **Mall Customers Dataset** to answer:
- How can we group customers mathematically based on their Income and Spending Score?
- What is the "optimal" number of segments (clusters) for this business?
- How do we visualize high-dimensional data (Age, Income, Spending) on a 2D screen?
- What specific marketing strategies should be applied to each unique group?

---

## **Approach**

### ***1. Environment Setup***
Imported `pandas` and `numpy` for data handling. For visualization, `matplotlib` and `seaborn` were used. Crucially, `sklearn` was used for the **K-Means** algorithm, scaling, and dimensionality reduction tools (**PCA** and **t-SNE**).

### ***2. Data Loading and Inspection***
- Downloaded the dataset directly from Kaggle using `kagglehub`.
- Used `.info()` and `.describe()` to understand the range of Age (18-70), Annual Income ($15k-$137k), and Spending Score (1-100).
- Confirmed the dataset is structured and clean, ready for mathematical modeling.

### ***3. Exploratory Data Analysis (EDA)***
- **Gender Distribution:** Created a countplot to see the balance between Male and Female shoppers.
- **Pairplots & KDE Plots:** Visualized the relationship between Income and Spending Score. The KDE plot specifically showed "hotspots" or density zones where most customers fall.
- **Feature Scaling:** Since K-Means uses "distance" to group data, we used `StandardScaler` to ensure that a high income value doesn't unfairly outweigh a smaller spending score.

### ***4. K-Means Clustering***
- **The "Noob" Explanation:** Imagine throwing a bunch of points on a floor and asking 5 robots (centroids) to move to the middle of the closest groups. That’s K-Means.
- **Elbow Method:** We ran the model for 1 to 10 clusters and plotted the **WCSS** (Within-Cluster Sum of Square).
- **Optimal K:** The "elbow" or the point where the graph stops dropping sharply was at **K=5**. This told us that 5 is the perfect number of customer segments.

### ***5. Dimensionality Reduction (PCA & t-SNE)***
- **The "Noob" Explanation:** We have 3 main features (Age, Income, Spend), but we live in a 2D world (our screens). **PCA (Principal Component Analysis)** and **t-SNE** act like a camera, taking a 3D situation and flattening it into a 2D picture while keeping the "clusters" visible.
- Applied PCA to reduce the 3D scaled data into 2 components for a clear scatter plot of the 5 segments.

---

## **Key Findings & Marketing Strategies**

Based on the 5 clusters identified, here are the proposed strategies:

1.  **Cluster: High Income, Low Spending (The "Careful" Group)**
    * *Strategy:* Send them "Premium" or "Investment" style advertisements. They have money but don't spend easily; they need high-value propositions.
2.  **Cluster: High Income, High Spending (The "Target" Group)**
    * *Strategy:* These are the VIPs. Give them early access to new collections, loyalty rewards, and personalized concierge services.
3.  **Cluster: Average Income, Average Spending (The "Standard" Group)**
    * *Strategy:* Use general marketing, "Buy 1 Get 1 Free" offers, and seasonal discounts to keep them coming back.
4.  **Cluster: Low Income, High Spending (The "Careless" Group)**
    * *Strategy:* Focus on trend-based marketing and low-cost "fast fashion" alerts. They are likely young and influenced by viral trends.
5.  **Cluster: Low Income, Low Spending (The "Budget" Group)**
    * *Strategy:* Offer extreme discounts, clearance sales, and "value for money" bundles to attract this price-sensitive segment.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | CSV loading and DataFrame management |
| `StandardScaler` | Scaling data so no single feature dominates |
| `KMeans` | The primary algorithm for creating customer segments |
| `PCA` | Linear dimensionality reduction for 2D visualization |
| `t-SNE` | Non-linear dimensionality reduction for complex cluster visualization |
| `seaborn` | Creating the beautiful KDE and Cluster scatter plots |