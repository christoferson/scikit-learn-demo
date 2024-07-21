import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

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


### Exploratory Data Analysis

# Histogram
fig = plt.figure()
used_car_price_df['MSRP'].hist(bins=42, figsize=(12, 5), color='b')
fig.suptitle('MSRP Histogram')
#plt.show()

# Heatmap - Null Values
sns.heatmap(used_car_price_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
#plt.show()

# Scatterplot
sns.scatterplot(x='Horsepower', y='MSRP', data=used_car_price_df)
#plt.show()

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
#plt.show()

# Plot figure to count Types
plt.figure(figsize=(16, 8))
sns.countplot(x = used_car_price_df['Origin'])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45)
#plt.show()

# Plot figure to count Types
plt.figure(figsize=(16, 8))
sns.countplot(x = used_car_price_df['DriveTrain'])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45)
#plt.show()

# Correlation Matrix
#correlation_matrix = used_car_price_df.corr()
#print(correlation_matrix)
#sns.heatmap(correlation_matrix, annot=True, cmap='YIGnBu')
#plt.show()

# Wordcloud Models
text = used_car_price_df.Model.values
stopwords = set(STOPWORDS)
wc = WordCloud(background_color = "black", max_words = 2000, max_font_size = 100, random_state = 3, 
              stopwords = stopwords, contour_width = 3).generate(str(text))  

fig = plt.figure(figsize = (18, 8))
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
#plt.show()


##### DATA PREPARATION ####

print(f"HEAD:\n {used_car_price_df.head()}")
# Perform One Hot Encoding for "Make", "Model", "Type", "Origin", "DriveTrain"
car_df = pd.get_dummies(used_car_price_df, columns=["Make", "Model", "Type", "Origin", "DriveTrain"])
print(f"HEAD:\n {car_df.head()}")


# Feeding input features to X and output (MSRP) to y
X = car_df.drop("MSRP", axis = 1)
y = car_df["MSRP"]

X = np.array(X)
y = np.array(y)

# Split the Dataset with 20% as test data

print("Split the Data")

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape} y_test.shape: {y_test.shape}")


# Train the Model
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
