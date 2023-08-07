import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1200, 1500, 2000, 1800, 2500],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Location': ['City Center', 'Suburb', 'City Center', 'Suburb', 'Outskirts'],
    'Price': [250000, 350000, 420000, 380000, 480000]
}

df = pd.DataFrame(data)

df['Location'] = df['Location'].map({'City Center': 0, 'Suburb': 1, 'Outskirts': 2})

X = df[['Area', 'Bedrooms', 'Location']]
y = df['Price']

lin_reg_model = LinearRegression()
lin_reg_model.fit(X, y)

if __name__ == "__main__":
    area = float(input("Enter the area of the new house (in sq. ft.): "))
    bedrooms = int(input("Enter the number of bedrooms in the new house: "))
    location = input("Enter the location of the new house (City Center/Suburb/Outskirts): ")
    location = 0 if location.lower() == 'city center' else 1 if location.lower() == 'suburb' else 2

    new_house_features = pd.DataFrame({'Area': [area], 'Bedrooms': [bedrooms], 'Location': [location]})
    predicted_price = lin_reg_model.predict(new_house_features)[0]

    print(f"The predicted price of the new house is ${predicted_price:.2f}")
