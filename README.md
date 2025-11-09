
# Medical Insurance Cost Prediction

## Project Overview

This project aims to build a machine learning model to predict individual medical insurance costs billed by health insurance companies. Accurate prediction of these costs can help insurers set better premiums, assist individuals in financial planning, and provide insights into the key factors driving healthcare expenses.

The project follows a standard data science pipeline: data loading and exploration, data preprocessing, model training with various algorithms, hyperparameter tuning, and model evaluation.

## Dataset

The dataset used is the **"Medical Cost Personal Datasets"**, a commonly used dataset for regression tasks.

**Source:** Likely from Kaggle or other public data repositories.

**Features:**
*   `age`: Age of the primary beneficiary (Integer)
*   `sex`: Gender of the beneficiary (String: 'male', 'female')
*   `bmi`: Body Mass Index, a key indicator of body weight relative to height (Float)
*   `children`: Number of children/dependents covered by the insurance (Integer)
*   `smoker`: Smoking status of the beneficiary (String: 'yes', 'no')
*   `region`: Residential area of the beneficiary in the US (String: 'northeast', 'northwest', 'southeast', 'southwest')

**Target Variable:**
*   `charges`: Individual medical costs billed by health insurance (Float)

## Project Structure

```
Medical_cost_insurance/
├── data/                           # Directory for dataset (if stored locally)
│   └── insurance.csv
├── notebooks/                      # Jupyter notebook for the main analysis
│   └── Medical_Insurance_Cost_Analysis.ipynb
├── src/                            # Source code for modular scripts (Potential future use)
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   └── model_training.py
├── models/                         # Saved models (Potential future use)
│   └── best_model.pkl
├── requirements.txt                # List of Python dependencies
├── README.md                       # Project documentation (this file)
└── .gitignore                     # Files and directories to ignore by Git
```

## Technologies Used

*   **Programming Language:** Python 3
*   **Libraries & Frameworks:**
    *   **Data Manipulation:** Pandas, NumPy
    *   **Data Visualization:** Matplotlib, Seaborn
    *   **Machine Learning:** Scikit-Learn
    *   **Model Interpretation:** SHAP (if used)

## Methodology

### 1. Data Loading and Exploratory Data Analysis (EDA)
*   Loaded the dataset and inspected its structure, checking for missing values and data types.
*   Performed univariate and bivariate analysis to understand the distribution of each feature and its relationship with the target variable (`charges`).
*   Key insights from EDA:
    *   `smoker` status has a very strong correlation with higher insurance charges.
    *   `bmi` and `age` also show a positive correlation with charges.
    *   The `charges` distribution is right-skewed.

### 2. Data Preprocessing
*   **Handling Categorical Variables:** Applied One-Hot Encoding to the `sex`, `smoker`, and `region` columns to convert them into a numerical format suitable for machine learning models.
*   **Feature Scaling:** Used `StandardScaler` to standardize numerical features (`age`, `bmi`, `children`) for models sensitive to feature scales (e.g., Linear Regression, SVM).

### 3. Model Building and Training
The dataset was split into training (80%) and testing (20%) sets. The following regression models were trained and evaluated:
*   **Linear Regression**
*   **Random Forest Regressor**
*   **Gradient Boosting Regressor**
*   **Support Vector Regressor (SVR)**
*   **XGBoost Regressor**

### 4. Model Evaluation
Models were evaluated using the following metrics:
*   **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
*   **Mean Absolute Error (MAE):** The average absolute difference between predictions and actual values.
*   **Root Mean Squared Error (RMSE):** The square root of the average of squared differences, penalizing larger errors more.

## Results

The tree-based ensemble models (Random Forest, Gradient Boosting, and XGBoost) consistently outperformed linear models.

**Best Performing Model:**
The **Gradient Boosting Regressor** (or **XGBoost**, depending on your final result) achieved the highest performance on the test set.

| Model                  | R² (Test) | MAE (Test)   | RMSE (Test)  |
| ---------------------- | --------- | ------------ | ------------ |
| Linear Regression      | ~0.78     | ~$4,200      | ~$5,800      |
| Random Forest          | ~0.87     | ~$2,600      | ~$4,500      |
| **Gradient Boosting**  | **~0.88** | **~$2,400**  | **~$4,300**  |
| XGBoost                | ~0.87     | ~$2,500      | ~$4,400      |

*Note: Replace the values above with the actual results from your notebook.*

## Key Findings & Insights

1.  **Smoking is the Single Biggest Factor:** The model confirms the EDA finding that being a smoker is the most significant predictor of high medical insurance costs.
2.  **Age and BMI are Critical:** As age and BMI increase, the predicted insurance charges also increase.
3.  **Non-Linear Relationships:** The superior performance of ensemble models over Linear Regression suggests the presence of complex, non-linear interactions between the features (e.g., the combined effect of high BMI and smoking).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sankyyy28/Medical_cost_insurance.git
    cd Medical_cost_insurance
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    Launch Jupyter Lab or Notebook and open the `Medical_Insurance_Cost_Analysis.ipynb` file in the `notebooks/` directory. Execute the cells sequentially to reproduce the analysis.

## Future Work

*   **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` more extensively to further optimize the best-performing model.
*   **Feature Engineering:** Create new features, such as interaction terms (e.g., `age * bmi` or a flag for `smoker_with_high_bmi`).
*   **Gather More Data:** A larger dataset with additional features (e.g., pre-existing conditions, family medical history, exercise habits) could significantly improve model accuracy.
*   **Deploy the Model:** Create a simple web application using Flask or Streamlit to allow users to input their details and get a cost prediction.

*   **Sanket** - [sankyyy28](https://github.com/sankyyy28)

---
