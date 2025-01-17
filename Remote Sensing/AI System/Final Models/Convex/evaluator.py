import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def calculate_metrics(y_true, y_pred):
    # Ensure y_true and y_pred are discrete class labels (integers)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1, recall, precision

def main():
    # Load the CSV containing results
    csv_file = "AI System/Final Models/Convex/results.csv"
    df = pd.read_csv(csv_file)
    
    # Ground truth values are already integers
    y_true = df['Ground Truth Label'].values  # Ground truth labels
    
    # Prepare to collect results
    all_metrics = []

    # Loop through each predicted column and calculate metrics
    for column in df.columns:
        if column not in ['Ground Truth Label']:  # Skip the ground truth column
            # Convert 'Predicted' column values to string before splitting
            df['Predicted_Label'] = df[column].apply(lambda x: int(str(x).split('.')[0]))  # Get integer part as label
            df['Predicted_Confidence'] = df[column].apply(lambda x: float(str(x).split('.')[1]) / 1000)  # Extract confidence

            y_pred = df['Predicted_Label'].values    # Predicted labels
            
            # Calculate the classification metrics
            accuracy, f1, recall, precision = calculate_metrics(y_true, y_pred)
            
            # Store the metrics for this column/model
            all_metrics.append({
                "Model": column,  # Use the column name as the model name
                "Accuracy": accuracy * 100,
                "F1 Score": f1,
                "Recall": recall,
                "Precision": precision,
            })
    
    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("eval.csv", index=False)
    print("Metrics for all columns saved to eval.csv")

if __name__ == "__main__":
    main()
