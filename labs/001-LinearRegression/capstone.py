import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


#labs/001-LinearRegression Horse Power,Fuel Economy (MPG)
fuel_economy_df = pd.read_csv("labs/001-LinearRegression/data/FuelEconomy.csv")
print(f"Head:\n {fuel_economy_df.head()}")
print(f"Tail:\n {fuel_economy_df.tail()}")
print(f"Describe:\n {fuel_economy_df.describe()}")
print(f"Info:\n {fuel_economy_df.info()}")

# Histogram
fig = plt.figure()
fuel_economy_df['Horse Power'].hist(bins=42, figsize=(12, 5), color='b')
fig.suptitle('Horse Power Histogram')
#plt.show()

# Histogram
fig = plt.figure()
fuel_economy_df['Fuel Economy (MPG)'].hist(bins=42, figsize=(12, 5), color='b')
fig.suptitle('Fuel Economy (MPG) Histogram')
#plt.show()

# Pairplot
sns.pairplot(fuel_economy_df)
#plt.show()

# Correlation Matrix
correlation_matrix = fuel_economy_df.corr()
sns.heatmap(correlation_matrix, annot=True)
#plt.show()


# JoinPlot
sns.jointplot(x ='Horse Power', y='Fuel Economy (MPG)', data=fuel_economy_df)
plt.show()

# LMPPlot
sns.lmplot(x ='Horse Power', y='Fuel Economy (MPG)', data=fuel_economy_df)
plt.show()

X = fuel_economy_df[['Horse Power']]
y = fuel_economy_df[['Fuel Economy (MPG)']]

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# Split the Dataset with 20% as test data

print("Split the Data")

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape} y_test.shape: {y_test.shape}")


# Train the Model
print("Train the Model")

regression_model_sklearn = LinearRegression(fit_intercept=True) #fit_intercept enable y intercept
regression_model_sklearn.fit(X_train, y_train)

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"regression_model_sklearn_accuracy={regression_model_sklearn_accuracy}")

print(f"coefficient={regression_model_sklearn.coef_}")
print(f"intercept={regression_model_sklearn.intercept_}")


# Plot the training data
y_predict = regression_model_sklearn.predict(X_train)
plt.figure(figsize = (10, 6))
plt.scatter(X_train, y_train, color = 'b')
plt.plot(X_train, y_predict, color = 'red')
plt.ylabel('Fuel Economy')
plt.xlabel('Horse Power (Train)')
plt.title('Fuel Economy vs. Horse Power')
plt.grid()
#plt.show()

# Plot the test data
y_predict = regression_model_sklearn.predict(X_test)
print(f"y_predict {y_predict.shape} \n {y_predict}")
plt.figure(figsize = (10, 6))
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_predict, color = 'red')
plt.ylabel('Fuel Economy')
plt.xlabel('Horse Power (Test)')
plt.title('Fuel Economy vs. Horse Power')
plt.grid()
plt.show()

# Predict X = 250 and X =320
X_input = np.array([[250], [320]]) #pd.DataFrame({'Temperature': [250, 10]})
y_predict = regression_model_sklearn.predict(X_input)
print(f"Predicted values:")
print(f"For 250 HP: {y_predict[0]}")
print(f"For 320 HP: {y_predict[1]}")


# Predict X = 240 and X =360
X_input = np.array([[240], [360]]) #pd.DataFrame({'Temperature': [240, 360]})
y_predict = regression_model_sklearn.predict(X_input)
print(f"Predicted values:")
print(f"For 240 HP: {y_predict[0]}")
print(f"For 360 HP: {y_predict[1]}")