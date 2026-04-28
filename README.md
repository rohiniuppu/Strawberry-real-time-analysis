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
  [ git clone https://github.com/rohiniuppu/Strawberry-real-time-analysis.git](https://strawberry-real-time-analysis-jx43pbgancbdf7cgugrgt2.streamlit.app/)
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
### Real-Time Dashboard
![Strawberry Dashboard]
<img width="1537" height="852" alt="image" src="https://github.com/user-attachments/assets/af887f83-9ef2-4a6e-b94a-c1c91f0618dc" />

### Model Performance
<img width="777" height="643" alt="image" src="https://github.com/user-attachments/assets/e38e5983-be8e-433c-860b-e44548709217" />

<img width="743" height="713" alt="image" src="https://github.com/user-attachments/assets/3496afb2-911c-4b1b-b4ca-30951ccde125" />




---
Developed for AI-based agricultural research.
