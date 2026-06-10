import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import joblib

# Load dataset
data = pd.read_csv("data.csv")

print(data)

# Visualization
plt.figure(figsize=(8,5))

plt.scatter(data['hours'], data['marks'])

plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.title("Study Hours vs Marks")

plt.grid(True)

plt.show()

# Input and Output
X = data[['hours']]
y = data['marks']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# Regression Line
plt.figure(figsize=(8,5))

plt.scatter(X, y)

plt.plot(X, model.predict(X), linewidth=2)

plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.title("Regression Line")

plt.grid(True)

plt.show()

# Save model
joblib.dump(model, "model.pkl")

print("Model Saved Successfully")