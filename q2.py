import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
data_dict = {
    "Ad_Spend": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Conversion": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 1 = Success, 0 = No Success
}

# Convert to DataFrame
df_ads = pd.DataFrame(data_dict)

# Split features and target
X_features = df_ads[["Ad_Spend"]]
y_target = df_ads["Conversion"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Prediction
new_ad_budget = [[95]]  # Ad Spend = $95
prediction_result = classifier.predict(new_ad_budget)

# Output Prediction
print(f"Predicted Conversion for Ad Spend {new_ad_budget[0][0]}: {'Yes' if prediction_result[0] == 1 else 'No'}")
