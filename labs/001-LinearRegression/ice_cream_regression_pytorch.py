import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Load and prepare data
icecream_sales_df = pd.read_csv("labs/001-LinearRegression/data/IceCreamData.csv")
#icecream_sales_with_max_temp_df = icecream_sales_df[icecream_sales_df['Temperature'] == icecream_sales_df['Temperature'].max()]
#print(f"record.temp_max:\n {icecream_sales_with_max_temp_df}")

# Convert DataFrame to numpy array first
X = icecream_sales_df[['Temperature']].values.astype('float32')
y = icecream_sales_df[['Revenue']].values.astype('float32')

# Normalize the data (this helps prevent numerical instabilities)
X_mean = X.mean()
X_std = X.std()
y_mean = y.mean()
y_std = y.std()

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Convert numpy arrays to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# Split the Dataset with 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, fit_intercept=True):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=fit_intercept)

    def forward(self, x):
        return self.linear(x)

# Training function
def train_model(model, X_train, y_train, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Create and train model with intercept
print("Train the Model (with intercept)")
model = LinearRegressionModel(input_dim=1, fit_intercept=True)

# Initialize the weights properly
with torch.no_grad():
    model.linear.weight.data.normal_(0, 0.01)
    if model.linear.bias is not None:
        model.linear.bias.data.zero_()

train_model(model, X_train, y_train, epochs=1000, lr=0.1)

# Get coefficients and intercept
with torch.no_grad():
    print(f"coefficient={model.linear.weight.data}")
    print(f"intercept={model.linear.bias.data}")

# Calculate R² score
def r2_score(y_true, y_pred):
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot

# Calculate accuracy (R² score)
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    accuracy = r2_score(y_test, y_pred_test)
    print(f"regression_model_pytorch_accuracy={accuracy.item()}")

# Function to denormalize predictions
def denormalize(x, mean, std):
    return x * std + mean

# Predict for X = 35 and X = 10
X_input = torch.FloatTensor([[35], [10]])
# Normalize the input
X_input_normalized = (X_input - X_mean) / X_std

with torch.no_grad():
    y_predict_normalized = model(X_input_normalized)
    # Denormalize the predictions
    y_predict = denormalize(y_predict_normalized, y_mean, y_std)
    print(f"Predicted values:")
    print(f"For 35 degrees: {y_predict[0].item():.2f}")
    print(f"For 10 degrees: {y_predict[1].item():.2f}")

# Plotting training data
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train)
    # Denormalize for plotting
    X_train_denorm = denormalize(X_train, X_mean, X_std)
    y_train_denorm = denormalize(y_train, y_mean, y_std)
    y_pred_train_denorm = denormalize(y_pred_train, y_mean, y_std)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_denorm, y_train_denorm, color='b')
    plt.plot(X_train_denorm, y_pred_train_denorm, color='red')
    plt.ylabel('Revenue [$]')
    plt.xlabel('Temperature [Celcius] (Train)')
    plt.title('Revenue vs. Temperature')
    plt.grid()
    plt.show()