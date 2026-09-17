# Car Price Prediction with Machine Learning

🔗 **Live Dashboard:** [Add after deployment]

## Problem Statement
Predict the fair selling price of a used car based on its specifications (original price, age, mileage, fuel type, transmission, and ownership history), using a supervised regression model trained on real used car listings.

## Objective
- Build and compare multiple regression models to predict used car selling prices
- Identify which factors most strongly influence resale value
- Understand where and why the model's predictions are less reliable
- Present the model through an interactive prediction dashboard

## Dataset
Source: [Car Price Prediction (Used Cars)](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars) (Kaggle)
- Original dataset: 301 rows, 9 columns, mixing both cars and motorcycles
- **Key data quality finding:** The dataset contained 101 motorcycle/scooter listings mixed with car listings (e.g., Royal Enfield, Honda Activa, TVS models). Since this task specifically targets car price prediction, motorcycles were identified and removed, leaving 200 car-only rows before further cleaning.
- After removing 2 exact duplicate rows: **198 clean rows** used for analysis and modeling
- Fields: Car_Name, Year, Selling_Price (target), Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner

## Technologies Used
- Python 3.10
- Pandas, NumPy — data manipulation
- Scikit-learn — machine learning (Linear Regression, Random Forest, Gradient Boosting)
- Matplotlib, Seaborn — visualization
- Streamlit — interactive prediction dashboard
- Joblib — model persistence

## Project Architecture
```
CodeAlpha_CarPricePrediction/
├── data/
│   ├── raw/                    # original dataset
│   └── processed/              # cleaned data, trained model, feature list
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_feature_engineering_modeling.ipynb
├── visualizations/              # exported charts
├── app.py                       # Streamlit prediction dashboard
├── requirements.txt
├── README.md
└── .gitignore
```
## Data Cleaning
- Identified and removed 101 motorcycle/scooter listings incorrectly mixed into the car dataset (verified by manually reviewing all 98 unique vehicle names)
- Removed 2 exact duplicate rows
- Verified no missing values across any column
- Confirmed categorical values (Fuel_Type, Selling_type, Transmission, Owner) were clean and consistent

## Feature Engineering
- Derived `Car_Age` from `Year` (2020 − Year) for more interpretable modeling
- Dropped `Car_Name` from model features (37 unique values, many appearing only once — risked overfitting; brand/model pricing signal is already captured indirectly through `Present_Price`)
- One-hot encoded categorical variables: Fuel_Type, Selling_type, Transmission

## Machine Learning Methodology
- Train/test split: 80/20, with `random_state=42` for reproducibility
- Compared three regression models: Linear Regression, Random Forest, Gradient Boosting
- Used 5-fold cross-validation to obtain a more reliable performance estimate than a single train/test split
- Performed hyperparameter tuning on the selected model using GridSearchCV

## Models Compared
| Model | MAE (Lakhs) | RMSE (Lakhs) | R² |
|---|---|---|---|
| Linear Regression | 1.42 | 2.62 | 0.797 |
| Random Forest | 1.10 | 2.65 | 0.792 |
| Gradient Boosting | 1.15 | 2.80 | 0.768 |

**Selected model: Random Forest** — chosen for its lower MAE and its ability to provide feature importance for explainability, despite a marginally lower R² than Linear Regression on this small dataset.

## Evaluation Metrics
- **MAE (Mean Absolute Error):** average magnitude of prediction error, in lakhs
- **RMSE (Root Mean Squared Error):** similar to MAE but penalizes large errors more
- **R² (coefficient of determination):** proportion of price variance explained by the model — not to be confused with classification accuracy

## Results
- **Final tuned Random Forest performance (test set):** MAE = 1.10 lakhs, RMSE = 2.59 lakhs, R² = 0.801
- **Cross-validated R²: 0.889** — a more reliable estimate than the single test-set split
- Hyperparameter tuning improved cross-validated R² from 0.852 (default) to 0.889 (tuned)

## Key Insights
- **Present_Price is overwhelmingly the strongest predictor** (80% feature importance), followed by Car_Age (14.4%) and Driven_kms (4.9%). Categorical features (fuel type, transmission, seller type, ownership) contributed less than 1% combined.
- **Error analysis revealed a single dominant outlier**: a Toyota Land Cruiser (present price ₹92.6 lakhs, the only luxury SUV in the dataset) accounted for a prediction error of -14.88 lakhs — by far the largest in the test set. Excluding this one outlier, MAE dropped from 1.10 to 0.75 lakhs, showing the model performs strongly on typical cars but struggles to generalize to rare, extreme-value listings outside its training distribution.
- **No meaningful relationship between Owner count and price** — despite the intuitive assumption that fewer previous owners means higher value, this feature had negligible importance in the model.

## Business/Real-World Impact
This model could support a used car marketplace by providing instant, data-driven price estimates for sellers and buyers, reducing reliance on manual appraisal for typical vehicles. The error analysis suggests such a tool should flag rare or high-value vehicles for manual review rather than fully automated pricing.

## Interactive Dashboard
An interactive Streamlit dashboard (`app.py`) allows users to input a car's specifications and receive an instant predicted selling price, along with a visual explanation of which factors drive the model's predictions.

## Installation
```bash
git clone https://github.com/sirivarshini7161/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## How to Run
- Notebooks: open `notebooks/01_data_understanding.ipynb` and `notebooks/02_feature_engineering_modeling.ipynb`
- Dashboard: `streamlit run app.py`

## Future Improvements
- Collect more data on rare/luxury vehicles to improve outlier prediction
- Experiment with log-transforming the target variable to reduce the influence of high-value outliers
- Add a model confidence interval alongside point predictions

## Author
Sirivarshini — B.Tech Data Science Student, CodeAlpha Data Science Internship