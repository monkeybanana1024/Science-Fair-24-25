import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_data(df):
    X = df[['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']].values
    y = df['Ground Truth Label']  # No remapping, directly using the ground truth values ranging from 0-100
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed, y

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Using a regression objective
        eval_metric='rmse',  # Root Mean Squared Error as the evaluation metric
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def make_predictions_and_save(model, X_test, y_test, output_file="predictions.csv"):
    predictions = model.predict(X_test)
    results = pd.DataFrame({
        "Ground Truth Label": y_test,
        "Predicted": predictions,
        "NDVI": X_test[:, 0],
        "Satellite": X_test[:, 1],
        "Slope": X_test[:, 2],
        "Topography": X_test[:, 3]
    })
    print(results)
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Plotting predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

def evaluate_model(model, X_test, y_test, X_train, feature_names):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', title="Feature Importance", xlabel="Weight")
    plt.show()
    
    # Plotting residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="r", linestyles="--", lw=2)
    plt.title("Residuals Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()
    
    # Plotting error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Error Distribution (Residuals)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(df):
    # Calculate the correlation matrix
    correlation_matrix = df[['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain', 'Ground Truth Label']].corr()
    
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

def main():
    csv_file = "AI System/Final Models/Convex/predictions.csv"
    df = load_data(csv_file)
    X, y = preprocess_data(df)
    feature_names = ['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']
    
    # Plot correlation matrix before splitting data
    plot_correlation_matrix(df)
    
    # Adjusting train-test split ratio to be more reasonable (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    model = train_xgboost_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, X_train, feature_names)
    make_predictions_and_save(model, X_test, y_test, output_file="AI System/Final Models/Convex/results.csv")

if __name__ == "__main__":
    main()
