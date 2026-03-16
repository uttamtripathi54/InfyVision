# 📈 InfyVision

> A clean, premium stock price prediction dashboard for **Infosys (INFY.NS)** — powered by LSTM deep learning, GARCH volatility modelling, and Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- **LSTM Price Prediction** — Multi-feature sequence model (window = 60 days) trained on 5 years of historical data
- **Technical Indicators** — RSI, MACD, GARCH Volatility, Volume Change, Close/Open Ratio
- **Monte Carlo Risk Simulation** — 500 simulation runs over a configurable 30-day horizon with 95% confidence bands
- **Interactive Dashboard** — Built with Streamlit & Plotly; premium light theme with dot-grid background
- **Model Performance Tab** — MAE, MAPE, Direction Accuracy, RMSE, and feature correlation analysis

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Price Prediction** | Next-day LSTM forecast, recent price chart, and technical indicator overlays |
| **Model Performance** | Error metrics, model specs, test results, and feature importance bar chart |
| **Risk Analysis** | Monte Carlo simulation, VaR (95%), loss probability, and price distribution histogram |

---

## 🗂️ Project Structure

```
InfyVision/
├── app.py                  # Streamlit dashboard (main entry point)
├── InfyVision.ipynb        # Model training notebook
├── saved_models/
│   ├── infy_lstm_model.keras
│   ├── infy_scaler.pkl
│   └── infy_config.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/InfyVision.git
cd InfyVision
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
Open and run all cells in **`InfyVision.ipynb`**. This will generate the `saved_models/` directory with the trained LSTM model, scaler, and config.

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard UI |
| `yfinance` | Historical stock data |
| `tensorflow` / `keras` | LSTM model |
| `scikit-learn` | MinMaxScaler |
| `arch` | GARCH volatility modelling |
| `plotly` | Interactive charts |
| `pandas`, `numpy` | Data manipulation |

---

## 🧠 Model Details

- **Architecture:** LSTM (64 → 32 units) with Dropout regularisation
- **Input Features:** `Close`, `GARCH_Vol`, `RSI`, `MACD`, `Volume_Change`, `Close_Open_Ratio`
- **Window Size:** 60 trading days
- **Train / Test Split:** 80 / 20
- **Training:** Up to 12 epochs with early stopping (best val loss: ~0.0040)
- **Test RMSE:** ₹46.76 over 235 test samples

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. InfyVision's predictions are not financial advice. Do not make investment decisions based on model outputs. Past performance is not indicative of future results.

---

## 👤 Author

**Uttam Tripathi**  
Crafted with 💙 — InfyVision v1.0

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).