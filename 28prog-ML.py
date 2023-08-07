import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Purchases': [10, 5, 20, 15, 8, 25, 30, 12],
    'Returns': [2, 1, 3, 2, 1, 4, 5, 3],
    'TotalSpent': [100, 50, 200, 150, 80, 250, 300, 120]
}

df = pd.DataFrame(data)

X = df[['Purchases', 'Returns', 'TotalSpent']]

k = int(input("Enter the number of clusters (k): "))
kmeans_model = KMeans(n_clusters=k, random_state=42)
kmeans_model.fit(X)

if __name__ == "__main__":
    purchases = int(input("Enter the number of purchases made by the new customer: "))
    returns = int(input("Enter the number of returns made by the new customer: "))
    total_spent = int(input("Enter the total amount spent by the new customer: "))

    new_customer_features = [[purchases, returns, total_spent]]
    predicted_cluster = kmeans_model.predict(new_customer_features)[0]

    print(f"The new customer belongs to cluster {predicted_cluster}.")
