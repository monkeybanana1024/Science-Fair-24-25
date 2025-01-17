import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_data(df):
    X = df[['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']].values
    y = (df['Ground Truth Label'] // 5).clip(0, 24)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed, y

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=25,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def make_predictions_and_save(model, X_test, y_test, output_file="predictions.csv"):
    predictions = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)
    combined_predictions = []
    for i, prob in enumerate(predicted_probabilities):
        bucket = predictions[i]
        confidence = np.max(prob)
        combined_predictions.append(f"{bucket}.{int(confidence * 1000):03d}")
    results = pd.DataFrame({
        "Ground Truth Label": y_test,
        "Predicted": combined_predictions,
        "NDVI": X_test[:, 0],
        "Satellite": X_test[:, 1],
        "Slope": X_test[:, 2],
        "Topography": X_test[:, 3]
    })
    print(results)
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def evaluate_model(model, X_test, y_test, X_train, feature_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    feature_importance = model.feature_importances_
    for feature, importance in zip(feature_names, feature_importance):
        print(f"{feature}: {importance:.4f}")
    xgb.plot_importance(model, importance_type='weight', title="Feature Importance", xlabel="Weight")
    plt.show()

def main():
    csv_file = "AI System/Final Models/Convex/predictions.csv"
    df = load_data(csv_file)
    X, y = preprocess_data(df)
    feature_names = ['NDVI', 'Satellite', 'Slope', 'Topography', 'Rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_xgboost_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, X_train, feature_names)
    make_predictions_and_save(model, X_test, y_test, output_file="AI System/Final Models/Convex/results.csv")

if __name__ == "__main__":
    main()
