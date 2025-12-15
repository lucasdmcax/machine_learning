# Cars 4 You: Expediting Car Evaluations with Machine Learning

**Group 20 – Machine Learning Project**

## Team Members
| Name | Student Number |
|------|----------------|
| Gonçalo Torrão | 20250365 |
| João Paulo de Avila | 20250436 |
| Lucas Campos Ferreira | 20250448 |
| Maria Leonor Ribeiro | 20221898 |

**GitHub Repository:** [Cars 4 You – Machine Learning Project](https://github.com/lucasdmcax/machine_learning)

---

## Project Overview
Cars 4 You is an online car resale company that sells cars from multiple different brands. Their business model involves an online platform in which users who wish to sell their cars provide different sets of details about the car and send them to their chain of mechanics to get the car evaluated before purchasing it to, later, resell the car on a profit. Using this system, the managers were able to gather an extensive list of happy customers. However, the company’s growth has also led to increasing waiting lists for car inspection, which is driving potential customers to their competitors.

To address this, the company has reached out to us. Their main goal is to expedite the evaluation process by creating a predictive model capable of evaluating the price of a car based on the user’s input without needing the car to be taken to a mechanic.

---

## Objectives
- **Build and compare regression models** for car price prediction.
- **Evaluate model performance** using metrics such as MAE, RMSE, and R².
- **Optimize the best-performing model** through rigorous hyperparameter tuning and ablation analysis.
- **Deploy a scalable system** that can handle raw input data and generate predictions in real-time.

---

## Project Structure

```
.
├── Homework_Group20.ipynb    # Main project notebook (Analysis, Training, Prediction)
├── model_training_utils.py   # Helper module with preprocessing and CV logic
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Dataset folder
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data (features only)
│   ├── sample_submission.csv # Submission format
│   └── ...
```

### Key Files
*   **`Homework_Group20.ipynb`**: The central hub of the project. It contains the full narrative, from data exploration and business understanding to model benchmarking, advanced hyperparameter tuning, and final test set prediction.
*   **`model_training_utils.py`**: A custom utility library designed to ensure reproducibility and prevent data leakage. It handles:
    *   **Data Cleaning**: Outlier removal (IQR) and categorical standardization (Edit Distance).
    *   **Preprocessing**: Target Encoding, One-Hot Encoding, and Log Transformations.
    *   **Cross-Validation**: A custom `cross_validate_with_tuning` function that performs preprocessing *inside* each fold.

---

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lucasdmcax/machine_learning.git
    cd machine_learning
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    Open `Homework_Group20.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially.
    *   *Note: The notebook includes a `DEMO_MODE` flag. Set it to `False` to run the full, computationally intensive hyperparameter search.*

---

## Methodology: The "Ablation Analysis" Pipeline

We implemented a robust pipeline that treats preprocessing steps as hyperparameters. This allows us to empirically determine the best data transformation strategy for each model type.

### Key Preprocessing Steps
1.  **Categorical Standardization**: Uses Levenshtein distance to merge typos (e.g., "Mercedes-Benz" vs "Mercedes Benz").
2.  **Outlier Handling**: Clips or removes values exceeding  + $4 \times IQR$.
3.  **Target Encoding**: Encodes high-cardinality features (like `model`) based on the target mean, with smoothing to prevent overfitting.
4.  **Log Transformation**: Applies `log1p` to skewed features (`mileage`) and optionally to the target variable (`price`).

### Configuration Example (`model_config`)

The `cross_validate_with_tuning` function takes a `model_config` dictionary that defines the model class and the search space for both model hyperparameters and preprocessing flags.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import randint, uniform

# Example Configuration for HistGradientBoostingRegressor
hgb_config = {
    'model_class': HistGradientBoostingRegressor,
    'n_iter': 20,  # Number of random search iterations
    'param_distributions': {
        # Model Hyperparameters
        'learning_rate': uniform(0.01, 0.2),
        'max_iter': randint(100, 500),
        'max_depth': randint(3, 15),
        'l2_regularization': uniform(0, 10),
        
        # Preprocessing Hyperparameters (Ablation Analysis)
        'clean_outliers_flag': [True, False],      # Test with/without outlier cleaning
        'standardize_cats_flag': [True, False],    # Test with/without typo fixing
        'log_target': [True, False],               # Test with/without log-transforming price
        'feature_selection_k': [10, 20, 30, None]  # Test different feature subsets
    }
}

# Run the custom CV
results = cross_validate_with_tuning(
    X_train, 
    y_train, 
    cat_cols, 
    num_cols, 
    hgb_config,
    k=3
)
```

---

## Results

After training approximately **780 models** across various algorithms (Ridge, Lasso, Random Forest, Gradient Boosting, MLP), our analysis yielded the following insights:

*   **Best Model**: `HistGradientBoostingRegressor` achieved the lowest MAE on the validation set.
*   **Log-Transformation**: Critical for linear models (Ridge/Lasso) but showed mixed results for ensemble methods.
*   **Feature Importance**: `model` (Target Encoded), `age`, and `engineSize` were consistently the most predictive features.

The final model was retrained on the full training dataset using the optimal hyperparameter configuration before generating predictions for `test.csv`.
