# train_model.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

class SalesForecaster:
    def __init__(self, tune_hyperparameters_enabled=True):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.tune_hyperparameters_enabled = tune_hyperparameters_enabled

    def get_default_model_params(self):
        """Return default model parameters"""
        return {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }
    
    def load_data(self):
        """Load and validate the dataset"""
        try:
            # Get the path to the data file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, 'data', 'product_sales.csv')

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")

            # Load the data
            df = pd.read_csv(data_path)

            # Convert date column to datetime before validation
            df['date'] = pd.to_datetime(df['date'])

            # Validate the data
            self.validate_data(df)

            return df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the input data for required columns and data quality.

        Args:
            df (pd.DataFrame): Input DataFrame to validate

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("The input DataFrame is empty")

            # Check for required columns
            required_columns = ['date', 'store_id', 'product_id', 'sales', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                raise ValueError("Date column must be datetime type")

            # Check for null values in critical columns
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                print("Warning: Dataset contains missing values:")
                print(null_counts[null_counts > 0])

            # Validate data types
            expected_types = {
                'store_id': ['int64', 'int32'],
                'product_id': ['int64', 'int32'],
                'sales': ['float64', 'int64', 'int32'],
                'price': ['float64', 'int64', 'int32']
            }

            for column, expected_type in expected_types.items():
                if df[column].dtype.name not in expected_type:
                    print(f"Warning: Column {column} has type {df[column].dtype.name}, "
                        f"expected one of {expected_type}")

            # Check for negative values in sales and price
            if (df['sales'] < 0).any():
                raise ValueError("Dataset contains negative sales values")
            if (df['price'] < 0).any():
                raise ValueError("Dataset contains negative price values")

            # Check date range
            date_range = df['date'].max() - df['date'].min()
            if date_range.days < 365:
                print("Warning: Dataset spans less than a year, which might affect "
                    "seasonal feature generation")

            # Check for duplicate records
            duplicates = df.duplicated(subset=['date', 'store_id', 'product_id'])
            if duplicates.any():
                print(f"Warning: Found {duplicates.sum()} duplicate records")

            return True

        except Exception as e:
            print(f"Data validation error: {str(e)}")
            raise

    def split_data(self, X, y, test_size=0.2):
            """Split the data into training and test sets"""
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            return X_train, X_test, y_train, y_test

    def construct_model(self, X_train, y_train):
        """
        Construct the best model based on hyperparameter tuning flag
        """
        if self.tune_hyperparameters_enabled:  # Updated flag name
            print("\nTuning hyperparameters...")
            best_model = self.tune_hyperparameters(X_train, y_train)
            print("\nConstructing model with best hyperparameters...")
            return best_model
        else:
            print("\nConstructing model with default parameters...")
            return RandomForestRegressor(**self.get_default_model_params())
        
    def evaluate_model(self, X, y):
        """Evaluate the model on given data"""
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return mae, rmse, r2
    
    def create_features(self, df, is_training=True):
        """Create time series features."""
        df = df.copy()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Basic time features - using astype(np.int32) for integer features
        df['year'] = df['date'].dt.year.astype(np.int32)
        df['month'] = df['date'].dt.month.astype(np.int32)
        df['day_of_week'] = df['date'].dt.dayofweek.astype(np.int32)
        df['day_of_month'] = df['date'].dt.day.astype(np.int32)
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int32)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(np.int32)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(np.int32)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(np.int32)

        # Quarter features
        df['quarter'] = df['date'].dt.quarter.astype(np.int32)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(np.int32)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(np.int32)
        df['days_in_month'] = df['date'].dt.days_in_month.astype(np.int32)

        # Cyclical features - using float32 for continuous values
        df['sine_day_of_year'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25).astype(np.float32)
        df['cosine_day_of_year'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25).astype(np.float32)

        if is_training:
            # Lag features
            for lag in [1, 7, 14, 30]:
                df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'product_id'])['sales'].shift(lag).astype(np.float32)

            # Rolling mean features
            for window in [7, 14, 30, 60, 90]:
                df[f'sales_rolling_mean_{window}'] = (
                    df.groupby(['store_id', 'product_id'])['sales']
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                    .astype(np.float32)
                )

            # Year-over-year features
            df['sales_lag_365'] = (
                df.groupby(['store_id', 'product_id'])['sales']
                .shift(365)
                .astype(np.float32)
            )

            df['sales_rolling_mean_365'] = (
                df.groupby(['store_id', 'product_id'])['sales']
                .transform(lambda x: x.rolling(window=365, min_periods=1).mean())
                .astype(np.float32)
            )

            # Percentage change features
            for lag in [1, 7, 365]:
                pct_change = (
                    df.groupby(['store_id', 'product_id'])['sales']
                    .pct_change(periods=lag)
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(np.float32)
                )
                df[f'sales_pct_change_{lag}'] = pct_change

        # Ensure store_id and product_id are integers
        df['store_id'] = df['store_id'].astype(np.int32)
        df['product_id'] = df['product_id'].astype(np.int32)

        # Ensure price is float
        df['price'] = df['price'].astype(np.float32)

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
        for lag in [1, 7, 14, 30, 365]:
            future_df[f'sales_lag_{lag}'] = np.nan
        for window in [7, 14, 30, 60, 90, 365]:
            future_df[f'sales_rolling_mean_{window}'] = np.nan

        # Add percentage change columns
        for lag in [1, 7, 365]:
            future_df[f'sales_pct_change_{lag}'] = np.nan

        # Fill NaN values in lag features with the last known values from the training data
        for col in future_df.columns:
            if 'lag' in col or 'rolling_mean' in col or 'pct_change' in col:
                future_df[col] = df[col].iloc[-1]

        # Prepare features for prediction
        X_future = future_df[self.feature_columns]

        # Make predictions
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
            'is_weekend', 'is_month_start', 'is_month_end',
            'quarter', 'is_quarter_start', 'is_quarter_end', 'days_in_month',
            'sine_day_of_year', 'cosine_day_of_year'
        ]

        # Add lag and rolling mean features
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling_mean' in col or 'pct_change' in col]
        feature_cols.extend(lag_cols)

        # Store feature columns
        self.feature_columns = feature_cols

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['sales'].astype(np.float32)

        # Handle missing values
        X = X.ffill().bfill()

        # Ensure all numeric columns have the correct dtype
        for col in X.columns:
            if col in ['store_id', 'product_id', 'year', 'month', 'day_of_week', 
                    'day_of_month', 'week_of_year', 'is_weekend', 'is_month_start', 
                    'is_month_end', 'quarter', 'is_quarter_start', 'is_quarter_end', 
                    'days_in_month']:
                X[col] = X[col].astype(np.int32)
            else:
                X[col] = X[col].astype(np.float32)

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

        if self.tune_hyperparameters:
            # Initialize and train the model with hyperparameter tuning
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestRegressor(**self.get_default_model_params())
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
            'max_features': ['sqrt', 'log2', None],
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

            if self.tune_hyperparameters_enabled:  # Updated flag name
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(**self.get_default_model_params())

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
    forecaster = SalesForecaster(tune_hyperparameters_enabled=False)

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

    # Construct and train the model
    forecaster.model = forecaster.construct_model(X_train, y_train)
    forecaster.model.fit(X_train, y_train)

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