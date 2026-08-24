# 📦 M5 Walmart Demand Forecasting & Supply Chain Dashboard

An end-to-end Machine Learning pipeline and interactive web dashboard built to forecast daily demand and optimize inventory replenishment for Walmart stores.

---

## 🎯 Key Features
* **Time-Series Forecasting:** Evaluates XGBoost, LightGBM, SARIMA, Prophet, and Naive baselines for accurate daily sales predictions.
* **Inventory Policy Optimization:** Calculates dynamic **Safety Stock**, **Reorder Points (ROP)**, and **Economic Order Quantity (EOQ)** based on model residuals.
* **Interactive Dashboard:** Serves real-time interactive charts and supply chain policy recommendations using **Streamlit** and **Plotly**.

---

## 🏆 Model Evaluation Leaderboard

| Model | MAE | RMSE | MAPE (%) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **12.5** | **18.3** | **8.5%** | **🏆 Best Model** |
| LightGBM | 13.2 | 19.1 | 9.1% | Runner-up |
| Prophet | 15.3 | 22.1 | 10.2% | Evaluated |
| SARIMA | 18.7 | 25.4 | 12.8% | Evaluated |

---

## 💻 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Hakim824/walmart-demand-forecasting.git
cd walmart-demand-forecasting

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn xgboost lightgbm

# Run Streamlit dashboard
streamlit run app.py
