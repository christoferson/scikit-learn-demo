import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import shap

class SalesForecaster:
    
    def __init__(self, tune_hyperparameters=True):
        self.model = None
        self.feature_columns = None
        self.tune_hyperparameters = tune_hyperparameters

    def load_data(self):
        """Load and prepare the dataset"""
        # Get the path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'product_sales.csv')

        # Load the data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def create_features(self, df, is_training=True):
        df = df.copy()

        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Categorical features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

        # New features
        df['quarter'] = df['date'].dt.quarter
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['days_in_month'] = df['date'].dt.days_in_month
        df['sine_day_of_year'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['cosine_day_of_year'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        
        if is_training:
            # Lag features
            for lag in [1, 7, 14, 30, 365]:
                df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'product_id'])['sales'].shift(lag)

            # Rolling mean features
            for window in [7, 14, 30, 60, 90, 365]:
                df[f'sales_rolling_mean_{window}'] = df.groupby(['store_id', 'product_id'])['sales']\
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())

            # Percentage change features
            for lag in [1, 7, 365]:
                df[f'sales_pct_change_{lag}'] = df.groupby(['store_id', 'product_id'])['sales'].pct_change(periods=lag)

        return df

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
        y = df['sales'].copy()

        # Handle missing values
        X = X.bfill() #X = X.fillna(method='bfill')

        return X, y

    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def cross_validate(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        print("\nCross-validation results:")
        print(f"MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores) * 2:.2f})")
        print(f"RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores) * 2:.2f})")
        print(f"R2: {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores) * 2:.2f})")

        return mae_scores, rmse_scores, r2_scores

    def tune_hyperparameters_method(self, X, y):
        param_dist = {
            'num_leaves': [31, 62, 93],
            'max_depth': [-1, 5, 10, 15, 20],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        model = lgb.LGBMRegressor(random_state=42)

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                           n_iter=100, cv=5, random_state=42, n_jobs=-1,
                                           scoring='neg_mean_absolute_error')

        random_search.fit(X, y)

        print("Best parameters found: ", random_search.best_params_)
        print("Best MAE found: ", -random_search.best_score_)

        return random_search.best_estimator_

    def train_model(self, X, y):
        if self.tune_hyperparameters:
            print("\nTuning hyperparameters...")
            self.model = self.tune_hyperparameters_method(X, y)
        else:
            print("\nTraining model with default parameters...")
            self.model = lgb.LGBMRegressor(random_state=42)

        self.model.fit(X, y)
        return self.model

    def evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return mae, rmse, r2

    def forecast(self, df, days_to_forecast=30):
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, days_to_forecast+1)]

        future_df = pd.DataFrame({'date': future_dates})
        future_df['store_id'] = df['store_id'].iloc[-1]
        future_df['product_id'] = df['product_id'].iloc[-1]
        future_df['price'] = df['price'].iloc[-1]  # Assuming price stays constant

        future_df = self.create_features(future_df, is_training=False)

        for col in self.feature_columns:
            if col not in future_df.columns:
                future_df[col] = df[col].iloc[-1]

        X_future = future_df[self.feature_columns]

        future_sales = self.model.predict(X_future)

        future_df['predicted_sales'] = future_sales

        return future_df

    def plot_forecast(self, df, future_df):
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['sales'], label='Historical Sales')
        plt.plot(future_df['date'], future_df['predicted_sales'], label='Forecasted Sales')
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sales_forecast_lightgbm.png')
        plt.close()

    def plot_feature_importance(self, model, X):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance_lightgbm.png')
        plt.close()

        return importance

    def plot_shap_values(self, X):
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)

        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png')
        plt.close()

        # Plot detailed summary
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png')
        plt.close()

        # Return SHAP values for further analysis if needed
        return shap_values
        

def main(tune_hyperparameters=False):
    # Initialize the forecaster
    forecaster = SalesForecaster(tune_hyperparameters=tune_hyperparameters)

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

    print("\nTraining final model...")
    model = forecaster.train_model(X_train, y_train)

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
    importance = forecaster.plot_feature_importance(forecaster.model, X)
    print("\nFeature Importance:")
    print(importance)

    print("\nGenerating SHAP values...")
    shap_values = forecaster.plot_shap_values(X)

    print("\nSHAP plots have been saved as 'shap_feature_importance.png' and 'shap_summary_plot.png'")

    # If you want to analyze specific features or samples:
    print("\nSHAP values for the first prediction:")
    for i in range(min(5, len(X.columns))):  # Print first 5 features or less
        print(f"{X.columns[i]}: {shap_values[0][i]}")

if __name__ == "__main__":
    main(tune_hyperparameters=False)  # Set to False to disable hyperparameter tuning