#Assessment 3 HIT140 
#Group 57
"""notebook.py


"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
# Demographic data
dataset1 = pd.read_csv('Datasets/dataset1.csv')
# Screen time data
dataset2 = pd.read_csv('Datasets/dataset2.csv')
# Well-being scores data
dataset3 = pd.read_csv('Datasets/dataset3.csv')

# Ensure that the ID columns are correctly aligned in all datasets
print(f"Dataset 1 (Demographics) - ID Range: {dataset1['ID'].min()} to {dataset1['ID'].max()}")
print(f"Dataset 2 (Screen Time) - ID Range: {dataset2['ID'].min()} to {dataset2['ID'].max()}")
print(f"Dataset 3 (Well-being) - ID Range: {dataset3['ID'].min()} to {dataset3['ID'].max()}")

"""## 1. Data Cleaning

### 1.1 Merging Datasets
"""

# Merge dataset1 (Demographics) with dataset2 (Screen Time) on the 'ID' column
merged_data_1 = pd.merge(dataset1, dataset2, on='ID', how='inner')

# Merge the resulting dataset with dataset3 (Well-being) on the 'ID' column
merged_data = pd.merge(merged_data_1, dataset3, on='ID', how='inner')

print(merged_data.head())

# Summary statistics for the merged dataset
merged_data.describe().T

# Info on the merged dataset
print(merged_data.info())

"""### 1.2 Handling Missing Values"""

# Step 1: Handle Missing Values
# Check for any missing values in the dataset
print("Missing values in each column:")
print(merged_data.isnull().sum())

"""## 2. Feature Selection

### 2.1 Selecting Relevant Features
"""

# Step 3: Feature Selection (Based on the project objective)
# We are predicting well-being based on screen time, so we keep the relevant columns:
# ID, demographic features, screen time (from dataset2), and well-being scores (from dataset3)
# Relevant features:
selected_columns = [
    'ID', 'gender', 'minority', 'deprived',  # Demographic data
    'C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk',  # Screen time data
    'Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf',
    'Mkmind', 'Loved', 'Intthg', 'Cheer'  # Well-being scores
]

# Keep only the relevant features for the analysis
final_data = merged_data[selected_columns]

# Step 4: Check the final cleaned dataset
print("Cleaned Dataset:")
print(final_data.head())
print(final_data.info())

"""## 3. Exploratory Data Analysis (EDA)

"""

# Set plot aesthetics
sns.set(style="whitegrid")

"""
###  3.1 Distribution of Screen Time"""

# Plot the distribution of screen time on weekdays and weekends for all types of activities
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

plt.figure(figsize=(12, 8))
for col in screen_time_cols:
    sns.histplot(final_data[col], bins=20, kde=True, label=col)

plt.title('Distribution of Screen Time (Weekdays vs Weekends)', fontsize=16)
plt.xlabel('Hours per Day')
plt.ylabel('Count')
plt.legend(title="Screen Time Variables")
plt.show()

"""### 3.2 Distribution of Well-Being Scores"""

# Plot the distribution of well-being scores
well_being_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

plt.figure(figsize=(12, 8))
for col in well_being_cols:
    sns.histplot(final_data[col], bins=5, kde=False, label=col)

plt.title('Distribution of Well-Being Scores', fontsize=16)
plt.xlabel('Score')
plt.ylabel('Count')
plt.legend(title="Well-Being Variables")
plt.show()

"""### 3.3 Correlation Heatmap Between Screen Time and Well-Being Indicators"""

# Calculate correlations between screen time and well-being indicators
correlation_matrix = final_data[screen_time_cols + well_being_cols].corr()

# Plot the heatmap of correlations
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Screen Time and Well-Being Indicators', fontsize=16)
plt.show()

"""### 3.4 Correlation Between Total Screen Time and Well-Being"""

# Create a new column for total screen time (weekdays + weekends)
final_data['total_screen_time'] = final_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)

# Plot correlation between total screen time and well-being indicators
plt.figure(figsize=(12, 8))
sns.heatmap(final_data[['total_screen_time'] + well_being_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Between Total Screen Time and Well-Being Scores', fontsize=16)
plt.show()

"""## 4. Data Wrangling

### 4.1 Create New Features
"""

# Create new features for total screen time on weekends and weekdays

# Total screen time on weekends (sum of screen time for computers, gaming, smartphones, and TV on weekends)
final_data['total_screen_time_we'] = final_data['C_we'] + final_data['G_we'] + final_data['S_we'] + final_data['T_we']

# Total screen time on weekdays (sum of screen time for computers, gaming, smartphones, and TV on weekdays)
final_data['total_screen_time_wk'] = final_data['C_wk'] + final_data['G_wk'] + final_data['S_wk'] + final_data['T_wk']

# Create a combined feature for total screen time (weekdays + weekends)
final_data['total_screen_time'] = final_data['total_screen_time_we'] + final_data['total_screen_time_wk']

# Verify the new columns
print(final_data[['total_screen_time_we', 'total_screen_time_wk', 'total_screen_time']].head())

"""### 4.2 Data Standardization"""

# Select the columns to standardize (screen time features and well-being scores)
columns_to_standardize = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'total_screen_time_we', 'total_screen_time_wk', 'total_screen_time']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
final_data[columns_to_standardize] = scaler.fit_transform(final_data[columns_to_standardize])

# Verify the standardized data
print(final_data[columns_to_standardize].head())

"""# 6. Inferential Analysis"""

import scipy.stats as stats

# Define a threshold for "high" and "low" weekend screen time (using median)
median_screen_time_we = final_data['total_screen_time_we'].median()

# Create two groups: high and low weekend screen time
high_screen_time_group = final_data[final_data['total_screen_time_we'] > median_screen_time_we]['Optm']
low_screen_time_group = final_data[final_data['total_screen_time_we'] <= median_screen_time_we]['Optm']

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(high_screen_time_group, low_screen_time_group)

print(f"T-test for Optimism based on high vs. low weekend screen time:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

"""## 5. Linear Regression Modelling

### 5.1 Simple Linear Regression
"""

# Predicting Optimism ('Optm') based on computer usage on weekends ('C_we')

# Define the feature (independent variable) and target (dependent variable)
X = final_data[['C_we']]  # Independent variable: screen time on weekends (C_we)
y = final_data['Optm']    # Dependent variable: Optimism (Optm)

# Initialize and fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Make predictions
y_pred = lin_reg.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Simple Linear Regression: Predicting Optimism based on C_we")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Print the coefficients
print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficient for C_we: {lin_reg.coef_[0]}")

"""### 5.2 Confidence Intervals for Multiple Linear Regression"""

# Calculating Confidence Intervals for Linear Regression Coefficients

# Predict and get the standard errors
y_pred_simple = lin_reg.predict(X)
n = len(X)
p = X.shape[1]

# Calculate residuals and standard error
residuals = y - y_pred_simple
rss = np.sum(residuals**2)
standard_error = np.sqrt(rss / (n - p - 1))

# Calculate standard errors for the coefficients
X_with_intercept = np.hstack([np.ones((n, 1)), X])
cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * standard_error**2
standard_errors = np.sqrt(np.diagonal(cov_matrix))

# Calculate 95% confidence intervals for coefficients
confidence_interval_95 = [
    (lin_reg.coef_[i] - 1.96 * standard_errors[i + 1],
     lin_reg.coef_[i] + 1.96 * standard_errors[i + 1])
    for i in range(p)
]

print(f"95% Confidence Interval for C_we coefficient: {confidence_interval_95}")

"""### 5.3 Multiple Linear Regression"""

# Define the features (independent variables) and target (dependent variable)
X_multi = final_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]  # Screen time variables
y_multi = final_data['Optm']  # Dependent variable: Optimism (Optm)

# Initialize and fit the multiple linear regression model
multi_reg = LinearRegression()
multi_reg.fit(X_multi, y_multi)

# Make predictions
y_multi_pred = multi_reg.predict(X_multi)

# Evaluate the multiple regression model
mse_multi = mean_squared_error(y_multi, y_multi_pred)
r2_multi = r2_score(y_multi, y_multi_pred)

print(f"Multiple Linear Regression: Predicting Optimism based on all screen time variables")
print(f"Mean Squared Error (MSE): {mse_multi}")
print(f"R-squared (R2 Score): {r2_multi}")

# Print the coefficients
print(f"Intercept: {multi_reg.intercept_}")
print(f"Coefficients: {multi_reg.coef_}")

"""### 5.4 Confidence Intervals for Multiple Linear Regression:"""

# Predict and get the standard errors
y_pred_multi = multi_reg.predict(X_multi)
n_multi = len(X_multi)
p_multi = X_multi.shape[1]

# Calculate residuals and standard error
residuals_multi = y_multi - y_pred_multi
rss_multi = np.sum(residuals_multi**2)
standard_error_multi = np.sqrt(rss_multi / (n_multi - p_multi - 1))

# Calculate standard errors for the coefficients
X_multi_with_intercept = np.hstack([np.ones((n_multi, 1)), X_multi])
cov_matrix_multi = np.linalg.inv(X_multi_with_intercept.T @ X_multi_with_intercept) * standard_error_multi**2
standard_errors_multi = np.sqrt(np.diagonal(cov_matrix_multi))

# Calculate 95% confidence intervals for coefficients
confidence_intervals_95_multi = [
    (multi_reg.coef_[i] - 1.96 * standard_errors_multi[i + 1],
     multi_reg.coef_[i] + 1.96 * standard_errors_multi[i + 1])
    for i in range(p_multi)
]

print(f"95% Confidence Intervals for Multiple Regression Coefficients: {confidence_intervals_95_multi}")