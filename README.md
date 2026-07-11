# ☀️ AI-Based Solar Power Generation Prediction Using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based web application that predicts **solar power generation (kW)** using weather and atmospheric parameters. The application helps estimate the expected power output of a solar power plant based on environmental conditions.

The model is trained using historical weather and solar power generation data and is deployed using **Flask**.

---

## 🚀 Features

- Predict solar power generation using weather parameters.
- Machine Learning model built using **XGBoost Regressor**.
- Data preprocessing and feature selection.
- User-friendly web interface built with HTML, CSS, and Flask.
- Fast and accurate predictions.
- Ready for deployment on Render.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- XGBoost

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Web Development
- Flask
- HTML
- CSS

### Model Deployment
- Joblib
- Git
- GitHub
- Render

---

## 📂 Project Structure

```
Solar-Power-Prediction/
│
├── model/
│   ├── xgboost_model.pkl
│   └── feature_names.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── dataset.zip
```

---

## 📊 Dataset

The dataset contains historical weather and solar power generation data.

### Target Variable

- generated_power_kw

### Selected Input Features

- Angle of Incidence
- Total Cloud Cover
- Zenith Angle
- Azimuth Angle
- Shortwave Radiation
- Total Precipitation
- Low Cloud Cover
- Wind Gust

---

## 🔍 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Model Saving
10. Flask Web Application
11. Deployment

---

## 🤖 Models Compared

- Linear Regression
- Decision Tree Regressor
- XGBoost Regressor ✅ (Best Model)

---

## 📈 Model Evaluation

The following metrics were used:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Among all the trained models, **XGBoost Regressor** achieved the best overall performance and was selected for deployment.

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/solar-power-prediction.git
```

Move into the project folder

```bash
cd solar-power-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 📷 Application Workflow

```
User Inputs Weather Parameters
            │
            ▼
Flask Web Application
            │
            ▼
Load Trained XGBoost Model
            │
            ▼
Predict Solar Power Generation
            │
            ▼
Display Prediction
```

---

## 🌍 Real-World Applications

- Solar Power Plants
- Smart Grid Systems
- Renewable Energy Forecasting
- Energy Management
- Sustainable Energy Planning
- Smart Cities

---

## 📌 Future Improvements

- Real-time weather API integration.
- Automatic weather data collection.
- Interactive dashboard.
- Prediction history.
- Mobile-friendly interface.
- Cloud deployment with Docker and CI/CD.

---

## 👩‍💻 Developed By

**Sneha**

Machine Learning & AI Enthusiast

---

## ⭐ Acknowledgement

This project was developed as part of an internship to demonstrate the complete Machine Learning workflow, including data preprocessing, model training, evaluation, web application development, and deployment.