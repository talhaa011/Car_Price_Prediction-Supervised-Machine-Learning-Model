# Car Price Prediction 🚗📈

This project focuses on predicting car prices using machine learning techniques. The model is trained on a dataset that includes various features of used cars. The goal is to provide a reliable estimation of a car's selling price based on input attributes.

## 📁 Project Structure

- `car-price-prediction.ipynb`: The Jupyter Notebook containing all the preprocessing, visualization, modeling, and evaluation steps.
- `README.md`: Project description and usage instructions.

## 📊 Dataset Features

| Column Name         | Description (can be customized)                |
|---------------------|------------------------------------------------|
| `car_ID`            | Unique identifier for each car                 |
| `symboling`         | Insurance risk rating                          |
| `CarName`           | Car brand and model name                       |
| `fueltype`          | Type of fuel used (e.g., gas, diesel)          |
| `aspiration`        | Aspiration type (standard or turbo)            |
| `doornumber`        | Number of doors                                |
| `carbody`           | Type of car body (sedan, hatchback, etc.)      |
| `drivewheel`        | Drive wheel type (fwd, rwd, 4wd)               |
| `enginelocation`    | Location of engine (front or rear)             |
| `wheelbase`         | Distance between front and rear wheels         |
| `carlength`         | Length of the car                              |
| `carwidth`          | Width of the car                               |
| `carheight`         | Height of the car                              |
| `curbweight`        | Weight of the car without occupants or cargo   |
| `enginetype`        | Type of engine (e.g., ohc, ohcv, rotor)        |
| `cylindernumber`    | Number of engine cylinders                     |
| `enginesize`        | Size of the engine                             |
| `fuelsystem`        | Type of fuel system (e.g., mpfi, 2bbl)         |
| `boreratio`         | Bore ratio of the engine                       |
| `stroke`            | Stroke value of the engine                     |
| `compressionratio`  | Engine compression ratio                       |
| `horsepower`        | Engine horsepower                              |
| `peakrpm`           | Maximum RPM of the engine                      |
| `citympg`           | Mileage in city (miles per gallon)             |
| `highwaympg`        | Mileage on highway (miles per gallon)          |
| `price`             | Price of the car (target variable)             |


The dataset is cleaned, and feature engineering is performed for better model performance.

## 🔍 Exploratory Data Analysis (EDA)

The goal of EDA is to uncover insights and relationships in the data that influence the car's price. The following steps were taken:

### 1. 🔧 Data Overview

- Checked dataset shape, column types, and null values.
- Found no missing values in the dataset.

### 2. 📊 Univariate Analysis

- **Price Distribution**: Car prices are right-skewed with a few high-end models.
- **Fuel Type**: Majority of cars use `gas`; a smaller portion uses `diesel`.
- **Aspiration**: Most cars have standard aspiration; some have turbo.
- **Car Body**: `sedan`, `hatchback`, and `wagon` are the most common types.
- **Drive Wheel**: `fwd` (front-wheel drive) dominates the dataset.
  Visualizations were created using:
- `seaborn` (heatmaps, bar plots, boxplots)
- `matplotlib` (distribution plots, histograms)
### 4. 🔁 Correlation Matrix

- Created a heatmap of correlations to identify multicollinearity.
- Key positively correlated features with `price`:
  - `enginesize`
  - `horsepower`
  - `curbweight`
  - `carwidth`

## 🧹 Data Preprocessing

Steps involved:

- Dropping irrelevant columns like `Car_Name`.
- Creating a new column `Current_Year` and deriving `Car_Age`.
- Encoding categorical variables using `pd.get_dummies()`.
- Feature scaling and train-test splitting.

## 🤖 Model Building

The following regression models were trained and evaluated:

- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Random Forest Regressor**

### 📌 Best Performing Model

The **Random Forest Regressor** achieved the highest accuracy and lowest error, making it the best choice for deployment.

## 📈 Model Evaluation

Evaluation metrics used:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Cross-Validation Scores**

The Random Forest model consistently showed the best performance across these metrics.

## 💾 Model Saving

The best model (Random Forest Regressor) was saved using the `pickle` library for future use in deployment or app integration.

```python
import pickle
file = open('random_forest_model.pkl','wb')
pickle.dump(model, file)
