# Weather Condition Check Code

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = pd.read_csv("/content/weatherHistory.csv")

# 2. Features and target
features = ["Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]
X = df[features]
y = df["Temperature (C)"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Function to check weather condition

def check_weather(humidity, windspeed, pressure):
    sample = pd.DataFrame([[humidity, windspeed, pressure]], 
                          columns=["Humidity", "Wind Speed (km/h)", "Pressure (millibars)"])
    temp = model.predict(sample)[0]

    if temp < 10:
        condition = "Cold"
    elif 10 <= temp <= 25:
        condition = "Pleasant"
    else:
        condition = "Hot"

    return round(temp, 2), condition


# Example checks
print(check_weather(0.80, 12, 1015))  
print(check_weather(0.40, 5, 1008))    
