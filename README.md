# Garment Employee Productivity Prediction

This project focuses on predicting the actual productivity of garment employees using the "Productivity Prediction of Garment Employees" dataset from the UCI Machine Learning Repository. The repository documents an evolution from a basic modeling approach to a robust, professional data science workflow, emphasizing feature engineering, advanced modeling techniques, and reliable evaluation methods.

-----

  * **Student:** [César Diego Ruelas Flores](https://www.linkedin.com/in/diego-ruelas-flores/)
  * **Program:** Big Data and Data Science
  * **Institution:** [TECSUP](https://www.tecsup.edu.pe/)
  * **Course:** Data Mining
  * **Date:** June 19, 2025

### Instructor

> **[Luis Paraguay Arzapalo](https://www.linkedin.com/in/luisparaguay/)**
>
> Systems Engineer and Master in Information Technology Management from ESAN and La Salle, Spain. Specialist in Big Data, Business Intelligence, Machine Learning, Cloud, SQL, Data Modeling, and Agility.

-----

## Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%231-project-overview)
2.  [Project Evolution: A Tale of Two Notebooks](https://www.google.com/search?q=%232-project-evolution-a-tale-of-two-notebooks)
      - [Initial Approach: `LAB14-RUELAS.ipynb`](https://www.google.com/search?q=%2321-initial-approach-lab14-ruelasipynb)
      - [Advanced Approach: `LAB14-RUELAS-robust.ipynb`](https://www.google.com/search?q=%2322-advanced-approach-lab14-ruelas-robustipynb)
      - [Interpreting the Final Results](https://www.google.com/search?q=%2323-interpreting-the-final-results)
3.  [Project Structure](https://www.google.com/search?q=%233-project-structure)
4.  [Methodology and Technical Details](https://www.google.com/search?q=%234-methodology-and-technical-details)
      - [Feature Engineering](https://www.google.com/search?q=%2341-feature-engineering)
      - [Modeling and Hyperparameter Tuning](https://www.google.com/search?q=%2342-modeling-and-hyperparameter-tuning)
      - [Evaluation](https://www.google.com/search?q=%2343-evaluation)
      - [Code Modularity and Testing](https://www.google.com/search?q=%2344-code-modularity-and-testing)
5.  [How to Run the Project](https://www.google.com/search?q=%235-how-to-run-the-project)
      - [Prerequisites](https://www.google.com/search?q=%2351-prerequisites)
      - [Step-by-Step Installation and Execution](https://www.google.com/search?q=%2352-step-by-step-installation-and-execution)
6.  [Final Results](https://www.google.com/search?q=%236-final-results)

## 1\. Project Overview

The primary objective of this project is to build a reliable machine learning model to predict the `actual_productivity` of garment factory employees. The project leverages a real-world dataset and explores various regression and ensemble techniques to achieve the best possible performance. A significant emphasis is placed not just on achieving a high performance score, but on ensuring the evaluation process is robust, reproducible, and that the final model's performance is trustworthy.

  - **Dataset:** [Productivity Prediction of Garment Employees (UCI)](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)
  - **Target Variable:** `actual_productivity`
  - **Core Task:** Regression

## 2\. Project Evolution: A Tale of Two Notebooks

This repository contains two distinct approaches to solving the problem, demonstrating a progression from a simple baseline to a sophisticated and robust pipeline.

### 2.1. Initial Approach: `LAB14-RUELAS.ipynb`

The first notebook served as a baseline to establish initial performance metrics. Its methodology was characterized by:

  - **Simple Preprocessing:** Basic handling of missing values and outliers, with the `date` column being dropped entirely.
  - **Basic Ensemble Models:** Implementation of standard `scikit-learn` ensembles like `VotingRegressor` and `BaggingRegressor` with fixed, non-optimized hyperparameters.
  - **Simple Evaluation:** A single 80/20 train-test split was used to evaluate the models.

This approach yielded a `Bagging` model as the top performer with an **R² Score of 0.4973**. While a good start, this score is not fully reliable as it's based on a single, potentially "lucky" data split.

### 2.2. Advanced Approach: `LAB14-RUELAS-robust.ipynb`

The second notebook was created to address the limitations of the first and implement a professional, industry-standard workflow. Key improvements include:

  - **Advanced Feature Engineering:** Instead of dropping the `date` column, it was leveraged to extract valuable features like `month`, `week_of_year`, and `day_of_week`. New ratio-based features (`incentive_per_target`, `smv_per_worker`) were also created to provide deeper context to the models.
  - **State-of-the-Art Models:** Introduced highly optimized gradient boosting libraries: `XGBoost` and `LightGBM`.
  - **Robust Hyperparameter Tuning:** Implemented `RandomizedSearchCV` to automatically search for the best combination of hyperparameters for each model, saving significant manual effort and improving performance.
  - **Robust Evaluation:** Utilized **k-fold cross-validation** (with k=5) during the search process. This ensures that the performance estimate is stable and not dependent on a single data split.

### 2.3. Interpreting the Final Results

The robust process identified **XGBoost** as the best-performing model. On the final, held-out test set, it achieved an **R² Score of 0.4819**.

At first glance, this may seem slightly lower than the initial result. However, this new score is **far more valuable and trustworthy**. The consistency between the cross-validation score (R² ≈ 0.499) and the final test score (R² ≈ 0.482) confirms that the model generalizes well to new, unseen data and is not overfitted. The "robust" project successfully produced a reliable model with a performance metric that can be trusted in a real-world scenario.

## 3\. Project Structure

The project is organized using a standard data science project structure to ensure modularity, clarity, and ease of maintenance.

```
Data_Mining-LAB14/
│
├── .venv/                      # Virtual environment directory (isolated dependencies)
├── notebooks/
│   ├── LAB14-RUELAS.ipynb      # Notebook with the initial, simple approach
│   └── LAB14-RUELAS-robust.ipynb # Notebook with the final, robust approach
│
├── src/
│   ├── __init__.py             # Makes 'src' a Python package
│   ├── utils.py                # Utility functions for the simple notebook
│   └── utils_robust.py         # Advanced utility functions for the robust notebook
│
├── tests/
│   ├── __init__.py             # Makes 'tests' a Python package
│   ├── test_utils.py           # Pytest tests for utils.py
│   └── test_utils_robust.py    # Pytest tests for utils_robust.py
│
├── .gitignore                  # Specifies files for Git to ignore
├── LICENSE                     # Project license file
├── README.md                   # This file: project documentation
└── requirements.txt            # List of all Python dependencies
```

  - **Module Imports:** The notebooks, located in the `notebooks/` directory, use `sys.path` manipulation to correctly import modules from the `src/` directory, which is located one level up. This is a standard practice for maintaining a clean project structure.

## 4\. Methodology and Technical Details

### 4.1. Feature Engineering

The robust approach (`utils_robust.py`) significantly enhanced the dataset by:

1.  **Parsing Dates:** Converting the `date` string column into a datetime object.
2.  **Extracting Time-Based Features:** Creating `month`, `week_of_year`, and `day_of_week` to capture temporal patterns.
3.  **Creating Domain-Specific Ratios:** Engineering new features like `incentive_per_target` to model relationships between existing variables.

### 4.2. Modeling and Hyperparameter Tuning

A `ColumnTransformer` and `Pipeline` from `scikit-learn` were used to create a unified preprocessing and modeling workflow.

  - **Preprocessing:** Numerical features were scaled using `StandardScaler`, and categorical features were encoded using `OneHotEncoder`.
  - **Hyperparameter Search:** `RandomizedSearchCV` was used to efficiently search through a predefined grid of parameters for `RandomForest`, `XGBoost`, and `LightGBM`, using 5-fold cross-validation.

### 4.3. Evaluation

The final evaluation was conducted on a held-out test set (20% of the data) that was not used during any training or tuning phase. The primary metric for model comparison was the **R-squared (R²)** score, supplemented by **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

### 4.4. Code Modularity and Testing

  - All core logic (data loading, feature engineering, model training) was encapsulated in functions within the `src/utils_robust.py` module. This promotes code reuse and makes the main notebook cleaner and more readable.
  - Unit tests were written using the `pytest` framework to validate the functionality of the data processing functions, ensuring their reliability and correctness.

## 5\. How to Run the Project

To replicate the results of this project, follow the steps below. This workflow, based on a terminal and a virtual environment, is the recommended industry standard.

### 5.1. Prerequisites

  - Git
  - Python 3.10 or higher

### 5.2. Step-by-Step Installation and Execution

1.  **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd Data_Mining-LAB14
    ```

2.  **Create and Activate a Virtual Environment:** This isolates the project's dependencies.

    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate it (on Windows)
    .\.venv\Scripts\activate

    # On macOS/Linux:
    # source .venv/bin/activate
    ```

3.  **Install All Dependencies:** The `requirements.txt` file contains all necessary libraries.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Unit Tests (Optional but Recommended):** From the activated terminal, run `pytest` to ensure all utility functions are working correctly.

    ```bash
    pytest
    ```

5.  **Launch Jupyter Lab and Run the Notebook:**

    ```bash
    jupyter lab
    ```

      - In the Jupyter Lab interface that opens in your browser, navigate to the `notebooks/` directory.
      - Open `LAB14-RUELAS-robust.ipynb`.
      - Execute the cells sequentially from top to bottom.

## 6\. Final Results

After a full execution of the robust pipeline, the best model was identified and evaluated on the unseen test set, yielding the following performance:

  - **Best Performing Model:** XGBoost
  - **Final R² Score (on Test Set):** 0.4819
  - **Final RMSE (on Test Set):** 0.1173
  - **Final MAE (on Test Set):** 0.0768