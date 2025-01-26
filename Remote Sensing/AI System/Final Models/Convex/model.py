import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Preprocess the data
def preprocess_data(df):
    X = df[['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']].values
    y = df['Ground Truth Label']  # Directly using the ground truth values ranging from 0-100
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed, y

# Train XGBoost model with sklearn compatibility
def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test, X_train, feature_names):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="r", linestyles="--", lw=2)
    plt.title("Residuals Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()

    # Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Error Distribution (Residuals)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

# SHAP feature analysis
def shap_feature_analysis(model, X_train, feature_names):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Summary plot
    shap.summary_plot(shap_values.values, X_train, feature_names=feature_names)

    # Print feature importance based on SHAP values
    feature_importance = np.mean(np.abs(shap_values.values), axis=0)
    print("Feature Importance based on SHAP values:")
    for feature, importance in zip(feature_names, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # Dependence plots for each feature
    for i, feature_name in enumerate(feature_names):
        shap.dependence_plot(i, shap_values.values, X_train, feature_names=feature_names)

# Plot partial dependence graphs
def plot_partial_dependence_graphs(model, X_train, feature_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=list(range(len(feature_names))),
        feature_names=feature_names,
        ax=ax
    )
    plt.suptitle("Partial Dependence Plots")
    plt.tight_layout()
    plt.show()

# Correlation matrix
def plot_correlation_matrix(df):
    correlation_matrix = df[['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain', 'Ground Truth Label']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

# Main function
def main():
    csv_file = "AI System/Final Models/Convex/predictions.csv"
    df = load_data(csv_file)
    X, y = preprocess_data(df)
    feature_names = ['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']
    
    # Correlation matrix
    plot_correlation_matrix(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train the model
    model = train_xgboost_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, X_train, feature_names)

    # Feature analysis with SHAP
    shap_feature_analysis(model, X_train, feature_names)
    
    # Partial dependence plots
    plot_partial_dependence_graphs(model, X_train, feature_names)

if __name__ == "__main__":
    main()
