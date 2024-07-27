import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb


#labs/001-LinearRegression
university_admission_df = pd.read_csv("labs/006-XgBoost/data/university_admission.csv")
#print(icecream_sales_df)
print(f"Head:\n {university_admission_df.head()}")
print(f"Tail:\n {university_admission_df.tail()}")
print(f"Describe:\n {university_admission_df.describe()}")
print(f"columns:\n {university_admission_df.columns}")
print(f"shape:\n {university_admission_df.shape}")
print(f"isnull().sum():\n {university_admission_df.isnull().sum()}")
university_admission_df = university_admission_df.dropna()

# Explore Data

# check if there are any Null values
sns.heatmap(university_admission_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()

university_admission_df.hist(bins = 30, figsize = (20,20), color = 'r');
plt.show()

for i in university_admission_df.columns:
    
  plt.figure(figsize = (13, 7))
  sns.scatterplot(x = i, y = 'Chance_of_Admission', hue = "University_Rating", hue_norm = (1,5), data = university_admission_df)
  plt.show()

# Create the Training and Test Data

X = university_admission_df.drop("Chance_of_Admission", axis = 1)
y = university_admission_df["Chance_of_Admission"]

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# reshaping the array from (1000,) to (1000, 1)
y = y.reshape(-1,1)

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")


# Split the Dataset with 20% as test data

print("Split the Data")

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape} y_test.shape: {y_test.shape}")


# Train the Model (Linear Regression)

print("Train the Model")

regression_model_sklearn = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 30, n_estimators = 100)
regression_model_sklearn.fit(X_train, y_train)

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"regression_model_sklearn_accuracy={regression_model_sklearn_accuracy}")

#print(f"coefficients=\n{regression_model_sklearn.coef_}")
#print(f"intercept={regression_model_sklearn.intercept_}")


# Predict
y_predict = regression_model_sklearn.predict(X_test)
print(f"y_predict: \n{y_predict}")

k = X_train.shape[1] # Number of independent variables
n = len(X_test)
print(f"k={k} n={n}")

# Caluculate Evaluation Metrics
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-( (1-r2)*(n-1) / (n-k-1))

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
