# train_model.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns8
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

class SalesForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None

    def load_data(self):
        """Load and prepare the dataset"""
        # Get the path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'product_sales.csv')

        # Load the data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def split_data(self, X, y, test_size=0.2):
            """Split the data into training and test sets"""
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            return X_train, X_test, y_train, y_test

    def evaluate_model(self, X, y):
        """Evaluate the model on given data"""
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return mae, rmse, r2
    
    def create_features(self, df, is_training=True):
        """Create time-series features based on date"""
        df = df.copy()

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Categorical features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

        if is_training:
            # Create lag features for each store-product combination
            for lag in [1, 7, 14, 30]:  # Previous day, week, 2 weeks, month
                df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'product_id'])['sales'].shift(lag)

            # Create rolling mean features
            for window in [7, 14, 30]:
                df[f'sales_rolling_mean_{window}'] = df.groupby(['store_id', 'product_id'])['sales']\
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        return df

    def forecast(self, df, days_to_forecast=30):
        """Forecast sales for the next specified number of days"""
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, days_to_forecast+1)]

        future_df = pd.DataFrame({'date': future_dates})
        future_df['store_id'] = df['store_id'].iloc[-1]
        future_df['product_id'] = df['product_id'].iloc[-1]
        future_df['price'] = df['price'].iloc[-1]  # Assuming price stays constant

        # Create features for future dates
        future_df = self.create_features(future_df, is_training=False)

        # Add lag and rolling mean columns with NaN values
        for lag in [1, 7, 14, 30]:
            future_df[f'sales_lag_{lag}'] = np.nan
        for window in [7, 14, 30]:
            future_df[f'sales_rolling_mean_{window}'] = np.nan

        # Fill NaN values in lag features with the last known values from the training data
        for col in future_df.columns:
            if 'lag' in col or 'rolling_mean' in col:
                future_df[col] = df[col].iloc[-1]

        # Prepare features for prediction
        X_future = future_df[self.feature_columns]

        # Make predictions using self.model
        future_sales = self.model.predict(X_future)

        # Add predictions to the future dataframe
        future_df['predicted_sales'] = future_sales

        return future_df





    def prepare_features(self, df):
        """Prepare final feature set and handle missing values"""
        # List of features to use in the model
        feature_cols = [
            'store_id', 'product_id', 'price',
            'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]

        # Add lag and rolling mean features
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling_mean' in col]
        feature_cols.extend(lag_cols)

        # Store feature columns
        self.feature_columns = feature_cols

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['sales'].copy()

        # Handle missing values
        X = X.bfill()  # Changed from fillna(method='bfill')

        return X, y

    def plot_feature_importance(self, model, X):
        """Plot feature importance"""
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()

        # Save the plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(current_dir, 'feature_importance.png'))
        plt.close()

    def train_model(self, X, y):
        """Train the Random Forest model"""
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        # Plot feature importance
        self.plot_feature_importance(self.model, X)

        return self.model

    def tune_hyperparameters(self, X, y):
        # Define the parameter grid
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None]  # Removed 'auto'
        }

        # Create a base model
        rf = RandomForestRegressor(random_state=42)

        # Instantiate the random search model
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                           n_iter=100, cv=5, random_state=42, n_jobs=-1,
                                           scoring='neg_mean_absolute_error')

        # Fit the random search model
        random_search.fit(X, y)

        print("Best parameters found: ", random_search.best_params_)
        print("Best MAE found: ", -random_search.best_score_)

        return random_search.best_estimator_


    def plot_forecast(self, df, future_df):
        """Plot historical sales and forecasted sales"""
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['sales'], label='Historical Sales')
        plt.plot(future_df['date'], future_df['predicted_sales'], label='Forecasted Sales')
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(current_dir, 'sales_forecast.png'))
        plt.close()

    def print_feature_importance(self, model, X):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(importance)

        return importance

    def cross_validate(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        print("\nCross-validation results:")
        print(f"MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores) * 2:.2f})")
        print(f"RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores) * 2:.2f})")
        print(f"R2: {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores) * 2:.2f})")

        return mae_scores, rmse_scores, r2_scores


def main():
    # Initialize the forecaster
    forecaster = SalesForecaster()

    # Load and prepare data
    print("Loading data...")
    df = forecaster.load_data()

    print("Creating features...")
    df = forecaster.create_features(df, is_training=True)

    print("Preparing features...")
    X, y = forecaster.prepare_features(df)

    print("Dataset shape:", X.shape)
    print("\nFeatures created:", X.columns.tolist())

    # Split data into train and test sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = forecaster.split_data(X, y, test_size=0.2)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("\nPerforming cross-validation on training set...")
    mae_scores, rmse_scores, r2_scores = forecaster.cross_validate(X_train, y_train, n_splits=5)

    print("\nTuning hyperparameters...")
    best_model = forecaster.tune_hyperparameters(X_train, y_train)

    print("\nTraining final model with best hyperparameters...")
    forecaster.model = best_model  # Set the best model as the forecaster's model
    forecaster.model.fit(X_train, y_train)  # Train the best model on training data

    # Evaluate on training set
    print("\nEvaluating model on training set:")
    train_mae, train_rmse, train_r2 = forecaster.evaluate_model(X_train, y_train)
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training R2 Score: {train_r2:.2f}")

    # Evaluate on test set
    print("\nEvaluating model on test set:")
    test_mae, test_rmse, test_r2 = forecaster.evaluate_model(X_test, y_test)
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test R2 Score: {test_r2:.2f}")

    print("\nGenerating forecast...")
    future_df = forecaster.forecast(df, days_to_forecast=30)

    print("\nPlotting forecast...")
    forecaster.plot_forecast(df, future_df)

    print("\nForecast for the next 30 days:")
    print(future_df[['date', 'predicted_sales']])

    # Print feature importance
    forecaster.print_feature_importance(forecaster.model, X)

if __name__ == "__main__":
    main()