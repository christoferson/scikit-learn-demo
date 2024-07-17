import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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