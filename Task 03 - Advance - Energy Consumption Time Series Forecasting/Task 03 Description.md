# **Task 03 - Energy Consumption Time Series Forecasting**

## **Problem Statement**

Energy companies need to know how much electricity will be used in the future to manage power grids effectively and prevent shortages. Predicting energy usage is challenging because it changes based on the time of day, the day of the week, and seasonal patterns. 

This task focuses on building a short-term forecasting pipeline using historical household power consumption data. The core challenges addressed are:
- How do we transform raw, messy timestamp data into a structured format for machine learning?
- How do we extract "hidden" features like the hour of the day or whether it's a weekend to help the model learn?
- Which model performs best for time-based data: a classical statistical model (ARIMA), a modern seasonal model (Prophet), or a high-performance machine learning algorithm (XGBoost)?
- How do we visually compare the "Actual" vs. "Forecasted" results to see the model's accuracy?

---

## **Approach**

### ***1. Environment Setup***
- Imported `pandas` for time-series manipulation and `numpy` for numerical operations.
- Utilized `matplotlib` and `seaborn` for generating temporal line plots.
- Loaded specialized forecasting libraries: `statsmodels` for ARIMA, `prophet` (by Facebook), and `xgboost` for regression.

### ***2. Data Loading, Cleaning, and Resampling***
- **Loading:** Downloaded the Household Power Consumption dataset via `kagglehub`.
- **Cleaning:** Handled missing values (marked as '?') by converting them to NaNs and ensuring numerical columns were correctly typed.
- **Resampling:** The raw data was very granular. We resampled it to an **Hourly ('H')** frequency, calculating the mean energy usage for each hour to make the patterns clearer and the computation faster.

### ***3. Feature Engineering***
- To help the models understand time, we extracted specific features from the index:
    - **Hour:** To capture daily cycles (e.g., higher usage in the evening).
    - **Day of Week:** To distinguish between weekday routines and weekend patterns.
    - **Month:** To account for seasonal changes in energy consumption.

### ***4. Modeling and Comparison***
Three distinct approaches were implemented and compared:
- **ARIMA (AutoRegressive Integrated Moving Average):**
    - *The "Noob" Explanation:* This is an "Old is Gold" statistical method. It looks at the past values and the past errors to guess the next value. It's like saying, "If I used a lot of power in the last 5 hours, I'll probably use a lot in the 6th hour too."
- **Prophet:**
    - *The "Noob" Explanation:* Developed by Facebook, this tool is like a "smart calendar." It automatically figures out holidays and weekly/yearly seasons without you telling it much. It's very robust to missing data.
- **XGBoost (Extreme Gradient Boosting):**
    - *The "Noob" Explanation:* This is a heavyweight machine learning champion. It uses a "forest" of decision trees to find complex, non-linear patterns in the time-based features we engineered.

### ***5. Evaluation and Visualization***
- Each model's performance was measured using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to see how many units of energy the predictions were "off" on average.
- A comprehensive comparison plot was generated, showing the **Actual Test Data** alongside the forecasts from all three models to see which line followed the real data most closely.

---

## **Key Findings**

- **Data Resampling is Critical:** Converting minute-by-minute data to hourly data made the trends much more visible and reduced noise.
- **Feature Importance:** Time-based features (like 'hour') significantly improved the performance of the XGBoost model, as energy usage is highly dependent on daily routines.
- **Model Comparison:** - While **ARIMA** provides a solid baseline, it can struggle with complex seasonal patterns.
    - **Prophet** is excellent for capturing the "rhythm" of the week.
    - **XGBoost** often provides the lowest error rates because it can handle more complex relationships between the features.
- The final visualization showed that the models were successful in following the general "peaks and valleys" of daily energy usage.

---

## **Tools and Libraries Used**

| Library | Purpose |
|---|---|
| `pandas` | Parsing dates and resampling time-series data |
| `statsmodels.tsa.arima.model` | Implementing the classical ARIMA forecasting model |
| `prophet` | Facebook’s library for seasonal time-series forecasting |
| `xgboost` | High-performance regression for time-based features |
| `sklearn.metrics` | Calculating MAE and RMSE for model evaluation |
| `matplotlib` | Visualizing actual vs. forecasted energy trends |