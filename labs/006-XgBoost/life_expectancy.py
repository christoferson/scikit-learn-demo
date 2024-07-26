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
life_expectancy_df = pd.read_csv("labs/006-XgBoost/data/Life_Expectancy_Data.csv")
#print(icecream_sales_df)
print(f"Head:\n {life_expectancy_df.head()}")
print(f"Tail:\n {life_expectancy_df.tail()}")
print(f"Describe:\n {life_expectancy_df.describe()}")
print(f"columns:\n {life_expectancy_df.columns}")
print(f"shape:\n {life_expectancy_df.shape}")
print(f"isnull().sum():\n {life_expectancy_df.isnull().sum()}")
life_expectancy_df = life_expectancy_df.dropna()

# Explore Data

print(f"----------------------------- Exploratory Analysis -----------------------------------")

# check if there are any Null values
sns.heatmap(life_expectancy_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()

life_expectancy_df.hist(bins = 30, figsize = (15,15), color = 'r');
plt.show()

# Note that there is space after 'Life expectancy '
sns.scatterplot(x = 'Income composition of resources', y = 'Life expectancy ', hue = 'Status', data = life_expectancy_df);
plt.show()

# Note that there is space after 'Life expectancy '
sns.scatterplot(x = 'Income composition of resources', y = 'Life expectancy ', data = life_expectancy_df);
plt.show()

# Note that there is space after 'Life expectancy '
sns.scatterplot(x = 'Schooling', y = 'Life expectancy ', data = life_expectancy_df);
plt.show()

# Note that there is space after 'Life expectancy '
sns.scatterplot(x = 'Schooling', y = 'Life expectancy ', hue = 'Status', data = life_expectancy_df);
plt.show()


# Plot the correlation matrix
plt.figure(figsize = (20,20))
life_expectancy_numeric_cols = life_expectancy_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = life_expectancy_df[life_expectancy_numeric_cols].corr()
sns.heatmap(corr_matrix, annot = True)

#### Clean Data

print(f"----------------------------- Clean Data -----------------------------------")

# Perform one-hot encoding
life_expectancy_df = pd.get_dummies(life_expectancy_df, columns = ['Status'])

# Check the number of null values for the columns having null values
life_expectancy_df.isnull().sum()[np.where(life_expectancy_df.isnull().sum() != 0)[0]]

# Since most of the are continous values we fill them with mean
life_expectancy_df = life_expectancy_df.apply(lambda x: x.fillna(x.mean()),axis=0)

life_expectancy_df.isnull().sum()[np.where(life_expectancy_df.isnull().sum() != 0)[0]]

# Create the Training and Test Data

print(f"----------------------------- Create Train and Test Data -----------------------------------")

X = life_expectancy_df.drop("Life expectancy ", axis = 1)
y = life_expectancy_df["Life expectancy "]

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


# Train the Model (XGBoost)

print(f"----------------------------- Train Model -----------------------------------")

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
print(f"----------------------------- Calculate Metrics -----------------------------------")

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-( (1-r2)*(n-1) / (n-k-1))

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
