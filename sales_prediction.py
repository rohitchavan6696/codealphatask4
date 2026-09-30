import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("advertising.csv")

# 2. Display data
print(df.head())

# 3. Remove missing values
df = df.dropna()

# 4. Select features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Create model
model = LinearRegression()

# 7. Train model
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 10. Display coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 11. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")
plt.show()