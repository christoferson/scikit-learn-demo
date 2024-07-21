import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error, accuracy_score
from math import sqrt


stock_df = pd.read_csv("labs/003-MultipleLinearRegression/data/s&p500_stock_data.csv")
#used_car_price_df.dropna()
print(f"Head:\n {stock_df.head()}")
print(f"Tail:\n {stock_df.tail()}")
print(f"Describe:\n {stock_df.describe()}")
print(f"Info:\n {stock_df.info()}")
print(f"Dtypes:\n {stock_df.dtypes}")


### Exploratory Data Analysis

sns.jointplot(x='Employment', y='S&P 500 Price', data=stock_df)

sns.jointplot(x='Interest Rates', y='S&P 500 Price',data=stock_df)

sns.pairplot(stock_df)

#plt.show()


##### DATA PREPARATION ####

# Feeding input features to X and output (MSRP) to y
X = stock_df[['Interest Rates', 'Employment']]
y = stock_df['S&P 500 Price']

X = np.array(X)
y = np.array(y)


# Split the Dataset with 20% as test data

print("Split the Data")

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape} y_test.shape: {y_test.shape}")


# Train the Model (Linear Regression)

print("Train the Model")

regression_model_sklearn = LinearRegression(fit_intercept=True) #fit_intercept enable y intercept
regression_model_sklearn.fit(X_train, y_train)

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"regression_model_sklearn_accuracy={regression_model_sklearn_accuracy}")

print(f"coefficients=\n{regression_model_sklearn.coef_}")
print(f"intercept={regression_model_sklearn.intercept_}")

# Predict
y_predict = regression_model_sklearn.predict(X_test)
print(f"y_predict: \n{y_predict}")