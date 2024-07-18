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
