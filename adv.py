import pandas as pd
import statsmodels.api as sm

# Load the dataset (assuming the dataset is named 'advertising.csv')
data = pd.read_csv('advertising.csv')

# Selecting independent variables (TV, Radio, Newspaper) and dependent variable (Sales)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Adding a constant to the model (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get Residual Standard Error (RSE)
RSE = (model.mse_resid) ** 0.5

# Get R-squared value
R_squared = model.rsquared

# Get F-statistic
F_statistic = model.fvalue

# Print the results
print("Residual Standard Error (RSE):", RSE)
print("R-squared (R²):", R_squared)
print("F-statistic:", F_statistic)

# Commenting on the values
print("\nInterpretation:")
print(f"RSE represents the average error in predictions. A lower RSE ({RSE:.2f}) indicates a better fit.")
print(f"R² = {R_squared:.3f} means {R_squared*100:.1f}% of the variance in Sales is explained by the advertising budgets.")
print(f"F-statistic ({F_statistic:.2f}) tests overall model significance. A high F-statistic indicates strong evidence that at least one predictor is useful.")
