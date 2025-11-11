# 🏠 Housing Price Prediction using Machine Learning

Predicting house prices using data science and machine learning techniques.  
This project aims to build an accurate regression model that can estimate house prices based on various features such as area, number of rooms, location, and other property details.

---

## 📘 Project Overview

Housing prices depend on multiple factors like area, number of rooms, condition, year built, location, and more.  
The goal of this project is to:

- Analyze and visualize housing data 🧠  
- Preprocess and engineer meaningful features ⚙️  
- Train different machine learning models (Linear Regression, Random Forest, XGBoost) 🤖  
- Evaluate models using metrics such as RMSE and R² 📈  
- Build and deploy a prediction API using Flask 🚀

---

## 📂 Folder Structure

```
housing-price-prediction/
├─ data/
│  ├─ raw/                # Original dataset (train.csv, test.csv)
│  └─ processed/          # Cleaned and processed data
├─ notebooks/
│  ├─ 01-EDA.ipynb        # Exploratory Data Analysis
│  └─ 02-Modeling.ipynb   # Model building and evaluation
├─ src/
│  ├─ data_processing.py  # Data cleaning and preprocessing
│  ├─ features.py         # Feature engineering
│  ├─ models.py           # Model training scripts
│  └─ train.py            # Main training pipeline
├─ models/
│  └─ final_model.joblib  # Trained and saved model
├─ app/
│  └─ app.py              # Flask API for prediction
├─ reports/
│  └─ figures/            # Visualizations and plots
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation
```

---

## 🧩 Dataset

- **Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Target Variable:** `SalePrice`
- **Size:** ~1460 rows × 80+ features

### Key Features:
- `OverallQual` – Overall material and finish quality  
- `GrLivArea` – Above grade (ground) living area square feet  
- `GarageCars` – Size of garage in car capacity  
- `TotalBsmtSF` – Total square feet of basement area  
- `FullBath` – Full bathrooms above grade  
- `YearBuilt` – Original construction date  

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Aviap0311/housing-price-prediction.git
cd housing-price-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # For Windows
# or
source venv/bin/activate       # For Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Building Steps

### Step 1: Data Preprocessing
- Handle missing values (median/mode imputation)
- Encode categorical variables (OneHot / Label Encoding)
- Feature scaling and transformation
- Remove outliers and handle skewness

### Step 2: Feature Engineering
- Create new features (e.g., `TotalSF`, `Age`)
- Log-transform `SalePrice` for normalization
- Combine correlated features

### Step 3: Model Training
Trained multiple models and compared performance:

| Model | RMSE | R² Score |
|--------|------|-----------|
| Linear Regression | 0.180 | 0.85 |
| Random Forest | 0.132 | 0.92 |
| XGBoost | **0.126** | **0.93** |

### Step 4: Model Saving
Best model (`XGBoost`) saved as `final_model.joblib`.

---

## 🚀 API Deployment (Flask App)

You can run the prediction API locally using Flask:

### Run Flask App
```bash
cd app
python app.py
```

### API Endpoint
- **URL:** `http://127.0.0.1:5000/predict`
- **Method:** POST  
- **Input (JSON example):**
```json
{
  "OverallQual": 8,
  "GrLivArea": 2000,
  "GarageCars": 2,
  "TotalBsmtSF": 1000,
  "FullBath": 2,
  "YearBuilt": 2005
}
```

- **Output:**
```json
{
  "prediction": 254876.43
}
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **RMSE** | Root Mean Squared Error – penalizes large deviations |
| **MAE** | Mean Absolute Error – average prediction error |
| **R² Score** | Variance explained by the model |

---

## 📈 Visualizations

- Correlation Heatmap  
- Feature Importance (RandomForest / XGBoost)  
- Actual vs Predicted Plot  
- Residual Distribution  

All plots are saved under `reports/figures/`.

---

## 💾 Technologies Used

| Category | Libraries |
|-----------|------------|
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost, catboost |
| Model Saving | joblib |
| Deployment | flask |

---

## 🧮 How to Use for Prediction

Once model is trained and app is running:
1. Open `app/app.py`
2. Send POST request to `/predict` with JSON data (using Postman or cURL)
3. Get instant price prediction

---

## 📦 Future Improvements

- 🗺️ Add geospatial data (distance from city center)
- 🏙️ Integrate real-time data via API
- 🌐 Deploy on cloud (AWS, Render, or Hugging Face Spaces)
- 📈 Add Streamlit dashboard for interactive use
- 🔁 Auto retraining pipeline with new data

---

## 👨‍💻 Author

**Avinash Pawar**  
📧 Email: [avinashpawar1010@gmail.com](mailto:avinashpawar1010@gmail.com)  
💻 GitHub: [Aviap0311](https://github.com/Aviap0311)

---

## 🏁 Conclusion

This project demonstrates how data preprocessing, feature engineering, and advanced ML algorithms can be used to predict house prices accurately.  
It covers the **entire ML lifecycle** — from data exploration to deployment 🚀

---

> ⭐ If you like this project, don’t forget to star the repo and share it!
