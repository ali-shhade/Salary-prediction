import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
df_sal = pd.read_csv('regession/Salary_Data.csv')
df_sal = df_sal.dropna()  # preprocessing

x = df_sal.iloc[:, :1]
y = df_sal.iloc[:, 1:]

# Dataset split 1 (with fixed random_state)
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Dataset split 2 (without fixed random_state)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    x, y, test_size=0.2
)

# Model 1
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_train_predicted = regressor.predict(X_train)
y_test_predicted = regressor.predict(X_test)

# Model 2
regressor2 = LinearRegression()
regressor2.fit(X_train2, Y_train2)

y_train_predicted2 = regressor2.predict(X_train2)
y_test_predicted2 = regressor2.predict(X_test2)

# Evaluation using MAE
train_mae = mean_absolute_error(Y_train, y_train_predicted)
test_mae = mean_absolute_error(Y_test, y_test_predicted)

train_mae2 = mean_absolute_error(Y_train2, y_train_predicted2)
test_mae2 = mean_absolute_error(Y_test2, y_test_predicted2)

print("Model 1:")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")

print("\nModel 2:")
print(f"Train MAE: {train_mae2:.2f}")
print(f"Test MAE: {test_mae2:.2f}")

# Visualization
fig, axe = plt.subplots(1, 2, figsize=(8, 4))

axe[0].scatter(X_train, Y_train, color='green')
axe[0].scatter(X_test, Y_test, color='blue')
axe[0].scatter(X_test, y_test_predicted, color='yellow', marker='X')
axe[0].plot(X_train, y_train_predicted, color='orange')

axe[1].scatter(X_train2, Y_train2, color='green')
axe[1].scatter(X_test2, Y_test2, color='blue')
axe[1].scatter(X_test2, y_test_predicted2, color='yellow', marker='X')
axe[1].plot(X_train2, y_train_predicted2, color='orange')

plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()