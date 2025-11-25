import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def general_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Perform general data cleaning on the DataFrame.
    
    This function handles logical inconsistencies and data quality issues that
    don't require statistical calculations (mean, median, etc.) to prevent data
    leakage between training and validation sets.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing car data with columns:
            Brand, model, year, transmission, fuelType, mileage, tax, mpg, 
            engineSize, paintQuality%, previousOwners, hasDamage.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame with logical inconsistencies resolved.
    """

    df = df.copy()

    # Set negative values to NaN for features that shouldn't be negative
    for col in ['previousOwners', 'mileage', 'tax', 'mpg', 'engineSize']:
        df.loc[df[col] < 0, col] = np.nan

    for col in ['Brand', 'model', 'transmission', 'fuelType']:
        df[col] = df[col].str.lower()
        df[col] = df[col].replace('', np.nan)

    # Remove decimal part from 'year'
    df['year'] = np.floor(df['year']).astype('Int64')

    # Remove decimal part from 'previousOwners'
    df['previousOwners'] = np.floor(df['previousOwners']).astype('Int64')

    # Ensure 'paintQuality%' is within 0-100
    df.loc[(df['paintQuality%'] < 0) | (df['paintQuality%'] > 100), 'paintQuality%'] = np.nan

    # Fill missing 'hasDamage' with 1
    df['hasDamage'] = df['hasDamage'].fillna(1).astype('Int64')

    return df

def standardize_categorical_col(series: pd.Series, 
                                standardised_cats: list[str], 
                                distance_threshold: int = 2) -> pd.Series:
    """Standardizes a categorical column using edit distance with a threshold.

    1. Maps values to a standard category if they are a likely typo
       (i.e., within the edit distance_threshold).
    2. Keeps values that are already in the standard list.
    3. Groups all other values that don't match into an 'other' bin.
    
    Args:
        series (pd.Series): The categorical column to standardize.
        standardised_cats (list[str]): The list of "good" categories to match against.
        distance_threshold (int): The max edit distance to consider something a typo.
                                  A value of 1 or 2 is recommended.
                                  
    Returns:
        pd.Series: The standardized categorical column.
    """
    
    # 1. Get all unique non-null values from the series
    unique_values = series.dropna().unique()
    
    # 2. Build the mapping dictionary
    mapping = {}
    
    for x in unique_values:
        x_str = str(x)
        
        # Check if it's already a perfect match
        if x_str in standardised_cats:
            mapping[x] = x_str
            continue

        # Find the closest match and its distance
        distances = [nltk.edit_distance(x_str, cat) for cat in standardised_cats]
        min_distance = np.min(distances)
        
        if min_distance <= distance_threshold:
            closest_cat = standardised_cats[np.argmin(distances)]
            mapping[x] = closest_cat
        else:
            mapping[x] = 'other'
            
    return series.map(mapping)

def get_categories_high_freq(series: pd.Series, percent_threshold: float = 0.02) -> list[str]:
    """Get categories that appear more than a dynamic percentage threshold.
    
    Args:
        series (pd.Series): The categorical series to analyze.
        percent_threshold (float): The minimum percentage of total rows a category
                                   must have to be included (e.g., 0.01 for 1%).
                                   
    Returns:
        list[str]: List of categories with frequency above the dynamic threshold.
    """
    
    # Calculate the dynamic count threshold based on the percentage
    dynamic_count_threshold = len(series) * percent_threshold
    
    value_counts = series.value_counts()
    
    # Use the *same logic* as before, but with the new dynamic threshold
    high_freq_cats = value_counts[value_counts > dynamic_count_threshold].index.tolist()
    
    return high_freq_cats

def calculate_upper_bound(series: pd.Series) -> float:
    """Calculates the upper outlier bound for a pandas Series."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q3 + (1.5 * IQR)

def clean_outliers(series: pd.Series, 
                   upper_bound: float,
                   lower_bound: float = 0.0, 
                   return_missing: bool = True) -> pd.Series:
    """Clean outliers in the Series based on specified bounds.

    This function clips values outside the specified bounds or sets them to NaN.
    
    Args:
        series (pd.Series): The input Series containing numerical data.
        lower_bound (float): The lower bound for valid values.
        upper_bound (float): The upper bound for valid values.
        return_missing (bool): If True, set out-of-bound values to NaN.
                              If False, clip values to the bounds.
    
    Returns:
        pd.Series: The cleaned Series with outliers handled.
    """
    cleaned = series.copy()
    
    if return_missing:
        # Set out-of-bound values to NaN
        cleaned[(cleaned < lower_bound) | (cleaned > upper_bound)] = np.nan
    else:
        # Clip values to the specified bounds
        cleaned = cleaned.clip(lower=lower_bound, upper=upper_bound)
    
    return cleaned

def preprocess_data(X, cat_cols, num_cols, artifacts=None, fit=True):
    """
    Preprocess data using consistent transformations.
    
    Args:
        X (pd.DataFrame): Features to preprocess
        cat_cols (list): Categorical column names
        num_cols (dict): Numerical column names with types
        artifacts (dict): Preprocessing artifacts (if fit=False)
        fit (bool): If True, fit transformers; if False, use provided artifacts
        
    Returns:
        tuple: (X_processed, artifacts) if fit=True, else X_processed
    """
    X = X.copy()
    continuous_cols = [col for col, var_type in num_cols.items() if var_type == 'continuous']
    
    if fit:
        # Fit preprocessing on training data
        high_freq_cats = {col: get_categories_high_freq(X[col]) for col in cat_cols}
        mileage_upper = X['mileage'].quantile(0.95)
        outlier_bounds = {col: calculate_upper_bound(X[col]) for col in continuous_cols}
        medians = {col: X[col].median() for col in num_cols}
        
        artifacts = {
            'high_freq_cats': high_freq_cats,
            'mileage_upper': mileage_upper,
            'outlier_bounds': outlier_bounds,
            'medians': medians,
            'cat_cols': cat_cols,
            'num_cols': num_cols
        }
    else:
        high_freq_cats = artifacts['high_freq_cats']
        mileage_upper = artifacts['mileage_upper']
        outlier_bounds = artifacts['outlier_bounds']
        medians = artifacts['medians']
    
    # 1. Categorical preprocessing
    for col in cat_cols:
        X[col] = standardize_categorical_col(X[col], high_freq_cats[col])
        X[col] = X[col].fillna('other')
    
    # 2. Numerical outliers
    X['mileage'] = clean_outliers(X['mileage'], mileage_upper, 0, return_missing=False)
    
    for col in continuous_cols:
        if col != 'mileage':
            X[col] = clean_outliers(X[col], outlier_bounds[col])
    
    # 3. Fill missing values
    for col in num_cols:
        X[col] = X[col].fillna(medians[col])
    
    # 4. One-hot encoding
    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_data = pd.DataFrame(
            ohe.fit_transform(X[cat_cols]),
            columns=ohe.get_feature_names_out(cat_cols),
            index=X.index
        )
        artifacts['encoder'] = ohe
    else:
        ohe_data = pd.DataFrame(
            artifacts['encoder'].transform(X[cat_cols]),
            columns=artifacts['encoder'].get_feature_names_out(cat_cols),
            index=X.index
        )
    
    X = pd.concat([X.drop(columns=cat_cols), ohe_data], axis=1)
    
    # 5. Normalize numerical features
    numerical_cols = list(num_cols.keys())
    if fit:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        artifacts['scaler'] = scaler
    else:
        X[numerical_cols] = artifacts['scaler'].transform(X[numerical_cols])
    
    return (X, artifacts) if fit else X

def sample_hyperparameters(param_distributions, n_iter, seed):
    """
    Sample hyperparameters from distributions.
    
    Args:
        param_distributions (dict): Dictionary with parameter names as keys and
                                   distributions/lists as values
        n_iter (int): Number of parameter combinations to sample
        seed (int): Random seed for reproducibility
        
    Returns:
        list[dict]: List of parameter dictionaries
    """
    np.random.seed(seed)
    param_list = []
    
    for i in range(n_iter):
        params = {}
        for param_name, param_values in param_distributions.items():
            # Check if it's a scipy distribution (has .rvs method)
            if hasattr(param_values, 'rvs'):
                params[param_name] = param_values.rvs(random_state=seed + i)
            # Check if it's a list (discrete choices)
            elif isinstance(param_values, list):
                params[param_name] = np.random.choice(param_values)
            else:
                raise ValueError(f"Unknown parameter type for {param_name}")
        
        param_list.append(params)
    
    return param_list

def cross_validate_with_tuning(X_raw, y_raw, cat_cols_list, num_cols_dict, model_config, k=3, seed=42):
    """
    Perform k-fold cross-validation with manual hyperparameter search.
    Preprocessing is done within each fold to prevent data leakage.
    
    Args:
        X_raw (pd.DataFrame): Raw training features (not preprocessed)
        y_raw (pd.Series): Raw training target (not log-transformed)
        cat_cols_list (list): List of categorical column names
        num_cols_dict (dict): Dictionary of numerical columns with types
        model_config (dict): Configuration dictionary with keys:
            - 'model_class': sklearn model class (e.g., Ridge, Lasso, RandomForestRegressor)
            - 'param_distributions': dict of parameter distributions for sampling
            - 'n_iter': number of parameter settings to sample (default: 20)
        k (int): Number of CV folds (default: 3)
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Results with best_params, best_estimator, CV scores, and preprocessing artifacts
    """
    # Setup CV
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Sample hyperparameters once (will be used across all folds)
    n_iter = model_config.get('n_iter', 20)
    param_combinations = sample_hyperparameters(
        model_config['param_distributions'], 
        n_iter, 
        seed
    )
    
    # Storage for results
    fold_results = []
    best_score_overall = float('inf')
    best_params_overall = None
    
    print(f"Starting {k}-Fold CV with {model_config['model_class'].__name__} ({n_iter} hyperparam combinations)...")
    
    # Perform manual CV with preprocessing in each fold
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_raw), 1):
        # Split data for this fold
        X_train_fold = X_raw.iloc[train_idx].copy()
        X_val_fold = X_raw.iloc[val_idx].copy()
        y_train_fold = y_raw.iloc[train_idx].copy()
        y_val_fold = y_raw.iloc[val_idx].copy()
        
        # Preprocess data for this fold
        X_train_processed, fold_artifacts = preprocess_data(
            X_train_fold, cat_cols_list, num_cols_dict, fit=True
        )
        X_val_processed = preprocess_data(
            X_val_fold, cat_cols_list, num_cols_dict, 
            artifacts=fold_artifacts, fit=False
        )
        
        # Log-transform target
        y_train_log = np.log1p(y_train_fold)
        
        # Hyperparameter tuning: try each parameter combination
        best_fold_score = float('inf')
        best_fold_params = None
        best_fold_model = None
        
        for params in param_combinations:
            # Train model with these parameters
            model = model_config['model_class'](**params)
            model.fit(X_train_processed, y_train_log)
            
            # Predict on validation fold
            y_val_pred_log = model.predict(X_val_processed)
            y_val_pred = np.expm1(y_val_pred_log)
            
            # Calculate MAE
            fold_mae = mean_absolute_error(y_val_fold, y_val_pred)
            
            # Track best for this fold
            if fold_mae < best_fold_score:
                best_fold_score = fold_mae
                best_fold_params = params
                best_fold_model = model
        
        # Calculate train performance for best model
        y_train_pred_log = best_fold_model.predict(X_train_processed)
        y_train_pred = np.expm1(y_train_pred_log)
        train_mae = mean_absolute_error(y_train_fold, y_train_pred)
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'best_params': best_fold_params,
            'train_mae': train_mae,
            'val_mae': best_fold_score,
            'best_model': best_fold_model,
            'artifacts': fold_artifacts
        })
        
        # Track overall best across all folds
        if best_fold_score < best_score_overall:
            best_score_overall = best_fold_score
            best_params_overall = best_fold_params
    
    # Calculate mean and std of CV scores
    cv_scores = [r['val_mae'] for r in fold_results]
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    # Print summary table
    n_models_fitted = n_iter * k
    print_cv_summary(fold_results, mean_cv_score, std_cv_score, best_score_overall, n_models_fitted)
    
    # Create an unfitted model instance with the best parameters
    best_model_unfitted = model_config['model_class'](**best_params_overall)
    
    return {
        'best_params': best_params_overall,
        'best_estimator': best_model_unfitted,
        'mean_cv_score': mean_cv_score,
        'std_cv_score': std_cv_score,
        'best_fold_score': best_score_overall,
        'fold_results': fold_results
    }

def print_cv_summary(fold_results, mean_score, std_score, best_score, n_models_fitted):
    """Prints a summary table of the cross-validation results."""
    print(f"Fitted {n_models_fitted} models in total.")
    print(f"{'Fold':<6} | {'Train MAE':<12} | {'Val MAE':<12}")
    print("-" * 36)
    for r in fold_results:
        print(f"{r['fold']:<6} | £{r['train_mae']:<10.2f} | £{r['val_mae']:<10.2f}")
    print("-" * 36)
    print(f"Mean CV MAE: £{mean_score:.2f} ± £{std_score:.2f}")
    print(f"Best Fold MAE: £{best_score:.2f}\n")

def preprocess_test_data(test_df, artifacts):
    """
    Preprocess test data using artifacts from training.
    
    Args:
        test_df (pd.DataFrame): Raw test dataframe
        artifacts (dict): Preprocessing artifacts from cross_validate_with_tuning
        
    Returns:
        pd.DataFrame: Preprocessed test data ready for prediction
    """
    # General cleaning
    test_cleaned = general_cleaning(test_df)
    
    # Apply preprocessing using artifacts
    test_processed = preprocess_data(
        test_cleaned, 
        artifacts['cat_cols'], 
        artifacts['num_cols'], 
        artifacts=artifacts, 
        fit=False
    )
    
    return test_processed
