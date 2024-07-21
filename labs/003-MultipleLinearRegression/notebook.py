import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

print(f"Numpy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

used_car_price_df = pd.read_csv("labs/003-MultipleLinearRegression/data/used_car_price.csv")
#used_car_price_df.dropna()
print(f"Head:\n {used_car_price_df.head()}")
print(f"Tail:\n {used_car_price_df.tail()}")
print(f"Describe:\n {used_car_price_df.describe()}")
print(f"Info:\n {used_car_price_df.info()}")
print(f"Dtypes:\n {used_car_price_df.dtypes}")
print(f"MSRP.min:\n {used_car_price_df['MSRP'].min()}")
print(f"Null.Sum:\n {used_car_price_df.isnull().sum()}")


# Histogram
fig = plt.figure()
used_car_price_df['MSRP'].hist(bins=42, figsize=(12, 5), color='b')
fig.suptitle('MSRP Histogram')
plt.show()

# Heatmap - Null Values
sns.heatmap(used_car_price_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()

# Scatterplot
sns.scatterplot(x='Horsepower', y='MSRP', data=used_car_price_df)
plt.show()

# List out the unique Types
unique_types = used_car_price_df.Type.unique()
print(f"UniqueTypes (Types): {unique_types}")

# List out the unique Make
unique_types = used_car_price_df.Make.unique()
print(f"UniqueTypes (Makes): {unique_types}")

# Pairplot
#sns.pairplot(used_car_price_df)
#plt.show()

# Plot figure to count Types
plt.figure(figsize=(16, 8))
sns.countplot(x = used_car_price_df['Type'])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45)
plt.show()

# Plot figure to count Types
plt.figure(figsize=(16, 8))
sns.countplot(x = used_car_price_df['Origin'])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45)
plt.show()

# Plot figure to count Types
plt.figure(figsize=(16, 8))
sns.countplot(x = used_car_price_df['DriveTrain'])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45)
plt.show()