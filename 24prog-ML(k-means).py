import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample medical condition dataset
data = {
    'Symptom1': [0, 1, 0, 1, 0, 1, 1, 0],
    'Symptom2': [1, 0, 1, 1, 0, 0, 1, 0],
    'Symptom3': [1, 0, 1, 1, 0, 1, 0, 1],
    'Condition': [1, 0, 1, 1, 0, 0, 1, 0]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Define features and target variable
X = df[['Symptom1', 'Symptom2', 'Symptom3']]
y = df['Condition']

# Create and fit the KNN classifier with k neighbors
k = int(input("Enter the value of k (number of neighbors): "))
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X, y)

if __name__ == "__main__":
    # Get user inputs for the new patient's symptoms
    symptom1 = int(input("Enter the value for Symptom1 (0 or 1): "))
    symptom2 = int(input("Enter the value for Symptom2 (0 or 1): "))
    symptom3 = int(input("Enter the value for Symptom3 (0 or 1): "))

    # Predict the medical condition of the new patient
    new_patient_features = [[symptom1, symptom2, symptom3]]
    predicted_condition = knn_classifier.predict(new_patient_features)[0]

    if predicted_condition == 1:
        print("The patient is predicted to have the medical condition.")
    else:
        print("The patient is predicted not to have the medical condition.")
