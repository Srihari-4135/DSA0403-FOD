import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample customer churn dataset
data = {
    'UsagePattern1': [0.1, 0.2, 0.4, 0.3, 0.5],
    'UsagePattern2': [0.3, 0.1, 0.2, 0.4, 0.6],
    'Age': [25, 30, 22, 28, 35],
    'Income': [40000, 60000, 35000, 45000, 70000],
    'Churn': [0, 1, 1, 0, 1]
}

# Create a DataFrame from the sample data with explicit column names
df = pd.DataFrame(data)

# Define features and target variable
X = df[['UsagePattern1', 'UsagePattern2', 'Age', 'Income']]
y = df['Churn']

# Create and fit the Logistic Regression model
log_reg_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg_model.fit(X, y)

if __name__ == "__main__":
    # Get user inputs for the new customer's features
    usage_pattern1 = float(input("Enter the usage pattern 1 of the new customer: "))
    usage_pattern2 = float(input("Enter the usage pattern 2 of the new customer: "))
    age = int(input("Enter the age of the new customer: "))
    income = int(input("Enter the income of the new customer: "))

    # Predict the churn status of the new customer
    new_customer_features = [[usage_pattern1, usage_pattern2, age, income]]
    predicted_churn = log_reg_model.predict(new_customer_features)[0]

    if predicted_churn == 1:
        print("The customer is predicted to churn.")
    else:
        print("The customer is predicted not to churn.")
