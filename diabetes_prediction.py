import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("B:/MLOps_Dataset/diabetes2.csv")

# Select relevant columns
df_filtered = df[['BMI', 'Glucose']].copy()

# Remove invalid rows
df_filtered = df_filtered[(df_filtered['BMI'] > 0) & (df_filtered['Glucose'] > 0)]

# Define features and target
X = df_filtered[['BMI']]
y = df_filtered['Glucose']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('Simple Linear Regression: BMI vs Glucose')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
