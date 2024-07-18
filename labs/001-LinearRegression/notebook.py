import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

#labs/001-LinearRegression
icecream_sales_df = pd.read_csv("labs/001-LinearRegression/data/IceCreamData.csv")
#print(icecream_sales_df)
print(f"Head:\n {icecream_sales_df.head()}")
print(f"Tail:\n {icecream_sales_df.tail()}")
print(f"Describe:\n {icecream_sales_df.describe()}")
print(f"Revenue.min:\n {icecream_sales_df['Revenue'].min()}")


icecream_sales_with_max_temp_df = icecream_sales_df[ icecream_sales_df['Temperature'] == icecream_sales_df['Temperature'].max() ]
print(f"record.temp_max:\n {icecream_sales_with_max_temp_df}")

# Histogram
fig = plt.figure()
icecream_sales_df['Temperature'].hist(bins=42, figsize=(12, 5), color='b')
fig.suptitle('Temperature Histogram')
plt.show()

# Pairplot
sns.pairplot(icecream_sales_df)
plt.show()

# Correlation Matrix
correlation_matrix = icecream_sales_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

X = icecream_sales_df[['Temperature']]
y = icecream_sales_df[['Revenue']]

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# Split the Dataset with 20% as test data

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=2)

# Train the Model
print("Train the Model")

regression_model_sklearn = LinearRegression(fit_intercept=True) #fit_intercept enable y intercept
regression_model_sklearn.fit(X_train, y_train)

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"regression_model_sklearn_accuracy={regression_model_sklearn_accuracy}")

print(f"coefficient={regression_model_sklearn.coef_}")
print(f"intercept={regression_model_sklearn.intercept_}")


# Train the Model (Disable y intercept - fit_intercept = False)
print("Train the Model (Disable y intercept - fit_intercept = False)")

regression_model_sklearn = LinearRegression(fit_intercept=False) #fit_intercept enable y intercept
regression_model_sklearn.fit(X_train, y_train)

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"regression_model_sklearn_accuracy={regression_model_sklearn_accuracy}")

print(f"coefficient={regression_model_sklearn.coef_}")
print(f"intercept={regression_model_sklearn.intercept_}")