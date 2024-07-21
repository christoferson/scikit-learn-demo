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