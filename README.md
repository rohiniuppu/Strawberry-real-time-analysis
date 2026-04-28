# 🍓 Strawberry AI: Real-Time Quality Analysis Dashboard

An AI-powered system for the **non-destructive analysis of strawberries** using hyperspectral imaging simulation. This project evaluates quality parameters like tartaric acid, pH value, and firmness to provide real-time "Pickable" or "UnPickable" verdicts.

## 🚀 Features
- **Real-Time Dashboard:** A Streamlit-based interface to simulate camera feeds and sensor data.
- **Hyperspectral Quality Assessment:** Analyzes 6 key attributes:
  - Tartaric Acid
  - pH Value
  - Soluble Salts
  - Firmness
  - Color Index
  - Size (mm)
- **Machine Learning Engine:** Uses a Random Forest Classifier with **~90.5% accuracy**.
- **Research Artifacts:** Automatically generates Confusion Matrices and Feature Importance graphs.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd strawberry-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### 1. Generate Dataset & Train Model
First, generate the synthetic hyperspectral data and train the AI model:
```bash
python generate_dataset.py
python app.py
```
This will create `model.joblib` and save performance graphs.

### 2. Launch the Dashboard
Run the real-time analysis interface:
```bash
streamlit run dashboard.py
```

## 📈 Model Performance
The current model achieves a high classification accuracy on pickable vs. unpickable strawberries, meeting the standards required for automated agricultural pipelines.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 90.5% |
| **Precision** | 0.92 |
| **Recall** | 0.92 |

## 🖼️ Screenshots


---
Developed for AI-based agricultural research.
