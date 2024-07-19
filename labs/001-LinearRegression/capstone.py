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
#print(f"Revenue.min:\n {fuel_economy_df['Revenue'].min()}")

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


X = fuel_economy_df[['Horse Power']]
y = fuel_economy_df[['Fuel Economy (MPG)']]

print(f"X: {X.shape} \n{X}")
print(f"y: {y.shape} \n{y}")