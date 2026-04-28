import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    dataset_path = r"c:\Users\rohin\OneDrive\Documents\strawberry\strawberry_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run generate_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Features mentioned by user
    X = df[["tartaric acid", "ph value", "soluble salts", "firmness", "color", "size"]]
    y = df["target"]
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    model.fit(X_train, y_train)
    
    # --- NEW: Save the model ---
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib successfully!")

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 40)
    print(f"Model Accuracy: {accuracy:.4f} ({(accuracy * 100):.1f}%)")
    print("=" * 40)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["UnPickable", "Pickable"]))

    # --- NEW: Generate Graphs for Research Report ---
    print("\nGenerating visual graphs for your research report...")
    
    # 1. Confusion Matrix (Full Dataset as requested)
    y_all_pred = model.predict(X)
    cm = confusion_matrix(y, y_all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["UnPickable", "Pickable"], yticklabels=["UnPickable", "Pickable"])
    plt.title("Confusion Matrix (Full Dataset)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.title("Feature Importance in Hyperspectral Strawberries")
    plt.xlabel("Importance Score")
    plt.ylabel("Quality Attributes")
    plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphs saved as 'confusion_matrix.png' and 'feature_importance.png'")

if __name__ == "__main__":
    train_and_evaluate()
