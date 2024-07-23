import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error, accuracy_score

#labs/001-LinearRegression
university_admission_df = pd.read_csv("labs/005-LinearRegression/data/university_admission.csv")
#print(icecream_sales_df)
print(f"Head:\n {university_admission_df.head()}")
print(f"Tail:\n {university_admission_df.tail()}")
print(f"Describe:\n {university_admission_df.describe()}")


X = university_admission_df.drop("Chance_of_Admission", axis = 1)
y = university_admission_df["Chance_of_Admission"]

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')



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

k = 13 # Number of independent variables
n = len(X_test)

# Caluculate Evaluation Metrics
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-( (1-r2)*(n-1) / (n-k-1))

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
