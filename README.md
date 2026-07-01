# Real Estate Geospatial Price Predictor

An end-to-end machine learning pipeline that predicts housing prices based on property features and geospatial data. This project bridges the gap between exploratory data science and production engineering by serving a trained Random Forest model through a RESTful API.

## Tech Stack
* **Data Processing:** Python, Pandas, GeoPandas, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Model Deployment:** FastAPI, Uvicorn, Joblib

## Architecture & Pipeline

### 1. Data Engineering & EDA
The pipeline ingests raw housing data and handles missing values via median imputation. A correlation matrix is generated to identify primary price drivers. 

### 2. Model Training
A Random Forest Regressor was chosen over standard Linear Regression because real estate prices rarely scale perfectly linearly with their features. The ensemble method automatically handles non-linear relationships and feature interactions (e.g., the combined effect of square footage and neighborhood). 
* The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* The final trained model is serialized and exported as `house_price_model.pkl`.

### 3. API Deployment (MLOps)
To make the model usable for frontend applications or mobile apps, the `.pkl` file is loaded into memory on application startup via FastAPI. This exposes the model behind a scalable POST endpoint, demonstrating standard MLOps deployment practices.

## Local Setup & Execution

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/real-estate-predictor.git](https://github.com/yourusername/real-estate-predictor.git)
cd real-estate-predictor
