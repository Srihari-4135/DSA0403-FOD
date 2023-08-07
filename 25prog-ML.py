import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

if __name__ == "__main__":

    sepal_length = float(input("Enter the sepal length of the new flower: "))
    sepal_width = float(input("Enter the sepal width of the new flower: "))
    petal_length = float(input("Enter the petal length of the new flower: "))
    petal_width = float(input("Enter the petal width of the new flower: "))

    new_flower_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_species = dt_classifier.predict(new_flower_features)[0]

    species_names = iris.target_names
    print(f"The predicted species of the new flower is {species_names[predicted_species]}.")
