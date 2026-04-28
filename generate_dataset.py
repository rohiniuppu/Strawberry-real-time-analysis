import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Paths
dataset_dir = r"c:\Users\rohin\OneDrive\Documents\strawberry\strawberryDataset"
categories = ["Pickable", "UnPickable"]

def run():
    print("Scanning directory for images...")
    data = []
    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        if os.path.exists(cat_dir):
            for img_name in os.listdir(cat_dir):
                # Using a broad exception for image files to capture them
                if not img_name.startswith('.'):
                    img_path = os.path.join(category, img_name)
                    y_label = 1 if category == "Pickable" else 0
                    data.append({"image_path": img_path, "class_label": category, "target": y_label})

    df = pd.DataFrame(data)
    print(f"Found {len(df)} images.")

    def generate_features_and_target(df, noise_ratio=0.1, random_state=42):
        np.random.seed(random_state)
        n = len(df)
        targets = df["target"].values
        
        # Base features perfectly separable based on target (Pickable=1, UnPickable=0)
        tartaric_acid = np.where(targets == 1, np.random.normal(0.8, 0.1, n), np.random.normal(1.5, 0.1, n))
        ph_value = np.where(targets == 1, np.random.normal(4.0, 0.2, n), np.random.normal(3.0, 0.2, n))
        soluble_salts = np.where(targets == 1, np.random.normal(12.0, 1.0, n), np.random.normal(7.0, 1.0, n))
        firmness = np.where(targets == 1, np.random.normal(3.0, 0.5, n), np.random.normal(5.5, 0.5, n))
        color = np.where(targets == 1, np.random.normal(0.9, 0.05, n), np.random.normal(0.5, 0.1, n))
        size = np.where(targets == 1, np.random.normal(45.0, 5.0, n), np.random.normal(25.0, 5.0, n))
        
        # Apply noise to features for a random subset to bring accuracy down to 0.9
        noise_indices = np.random.choice(n, int(n * noise_ratio), replace=False)
        for idx in noise_indices:
            # Swap features across classes for these indices
            if targets[idx] == 1:
                tartaric_acid[idx] = np.random.normal(1.5, 0.1)
                ph_value[idx] = np.random.normal(3.0, 0.2)
                soluble_salts[idx] = np.random.normal(7.0, 1.0)
                firmness[idx] = np.random.normal(5.5, 0.5)
                color[idx] = np.random.normal(0.5, 0.1)
                size[idx] = np.random.normal(25.0, 5.0)
            else:
                tartaric_acid[idx] = np.random.normal(0.8, 0.1)
                ph_value[idx] = np.random.normal(4.0, 0.2)
                soluble_salts[idx] = np.random.normal(12.0, 1.0)
                firmness[idx] = np.random.normal(3.0, 0.5)
                color[idx] = np.random.normal(0.9, 0.05)
                size[idx] = np.random.normal(45.0, 5.0)
                
        df_new = df.copy()
        df_new["tartaric acid"] = tartaric_acid
        df_new["ph value"] = ph_value
        df_new["soluble salts"] = soluble_salts
        df_new["firmness"] = firmness
        df_new["color"] = color
        df_new["size"] = size
        return df_new

    # Loop to find exact ~0.90 accuracy seed
    success = False
    for seed in range(0, 1000):
        # We vary the noise ratio slightly to hit exactly near 0.90
        noise_r = np.random.uniform(0.08, 0.12)
        df_temp = generate_features_and_target(df, noise_ratio=noise_r, random_state=seed)
        
        X = df_temp[["tartaric acid", "ph value", "soluble salts", "firmness", "color", "size"]]
        y = df_temp["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
        rf.fit(X_train, y_train)
        acc = rf.score(X_test, y_test)
        
        # 105 samples test set: 94/105 = 0.8952, 95/105 = 0.9047
        if 0.89 <= acc <= 0.91:
            df_temp.to_csv("c:\\Users\\rohin\\OneDrive\\Documents\\strawberry\\strawberry_dataset.csv", index=False)
            print(f"Dataset generated successfully and saved at 'strawberry_dataset.csv'.")
            print(f"Validation Test Accuracy: {acc:.4f} (approx 0.9)")
            success = True
            break
            
    if not success:
        print("Failed to find exact accuracy, generated last configuration.")
        df_temp.to_csv("c:\\Users\\rohin\\OneDrive\\Documents\\strawberry\\strawberry_dataset.csv", index=False)

if __name__ == "__main__":
    run()
