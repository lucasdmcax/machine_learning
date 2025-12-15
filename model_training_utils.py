import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression

@lru_cache(maxsize=None)
def cached_edit_distance(s1: str, s2: str) -> int:
    """
    Wrapper around nltk.edit_distance with caching to speed up repeated calls.
    Since the set of car models is finite, this avoids re-calculating the same
    distances millions of times during CV.
    """
    return nltk.edit_distance(s1, s2)

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

    # Set zero values to NaN for engineSize (likely missing data)
    df.loc[df['engineSize'] == 0, 'engineSize'] = np.nan

    for col in ['Brand', 'model', 'transmission', 'fuelType']:
        df[col] = df[col].str.lower()
        df[col] = df[col].replace('', np.nan)

    # Handle year/age transformation
    if 'year' in df.columns:
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
    3. Keeps other values as is (instead of grouping into 'other').
    
    Args:
        series (pd.Series): The categorical column to standardize.
        standardised_cats (list[str]): The list of "good" categories to match against.
        distance_threshold (int): The max edit distance to consider something a typo.
                                  A value of 1 or 2 is recommended.
                                  
    Returns:
        pd.Series: The standardized categorical column.
    """
    
    # If no standard categories provided, return original series
    if not standardised_cats:
        return series

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
        distances = [cached_edit_distance(x_str, cat) for cat in standardised_cats]
        min_distance = np.min(distances)
        
        if min_distance <= distance_threshold:
            closest_cat = standardised_cats[np.argmin(distances)]
            mapping[x] = closest_cat
        else:
            mapping[x] = x_str # Keep original
            
    return series.map(mapping)

def get_categories_high_freq(series: pd.Series, percent_threshold: float = 0.001) -> list[str]:
    """Get categories that appear more than a dynamic percentage threshold.
    
    Args:
        series (pd.Series): The categorical series to analyze.
        percent_threshold (float): The minimum percentage of total rows a category
                                   must have to be included (e.g., 0.001 for 0.1%).
                                   
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
    return Q3 + (4 * IQR)

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

def preprocess_data(X: pd.DataFrame, 
                    cat_cols: list[str], 
                    num_cols: dict[str, str], 
                    artifacts: dict | None = None, 
                    fit: bool = True,
                    y: pd.Series | None = None,
                    clean_outliers_flag: bool = True,
                    standardize_cats_flag: bool = True,
                    normalize_flag: bool = True) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Preprocess data using consistent transformations.
    
    Args:
        X (pd.DataFrame): Features to preprocess.
        cat_cols (list[str]): Categorical column names.
        num_cols (dict[str, str]): Numerical column names with types.
        artifacts (dict | None): Preprocessing artifacts (if fit=False).
        fit (bool): If True, fit transformers; if False, use provided artifacts.
        y (pd.Series | None): Target variable for Target Encoding (required if fit=True).
        clean_outliers_flag (bool): If True, perform outlier cleaning.
        standardize_cats_flag (bool): If True, perform categorical standardization.
        normalize_flag (bool): If True, perform numerical normalization.
        
    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict]: (X_processed, artifacts) if fit=True, else X_processed.
    """
    X = X.copy()
    
    # Handle year -> age transformation
    if 'year' in X.columns:
        X['age'] = 2020 - X['year']
        X = X.drop(columns=['year'])

    # Log transform mileage
    if 'mileage' in X.columns:
        X['mileage'] = np.log1p(X['mileage'])
        
    # Update num_cols to reflect the change from year to age
    processing_num_cols = num_cols.copy()
    if 'year' in processing_num_cols:
        processing_num_cols.pop('year')
        processing_num_cols['age'] = 'continuous'
    
    continuous_cols = [col for col, var_type in processing_num_cols.items() if var_type == 'continuous']
    
    # Identify columns for Target Encoding vs One-Hot Encoding
    te_cols = [col for col in cat_cols if col == 'model']
    ohe_cols = [col for col in cat_cols if col != 'model']
    
    if fit:
        # Fit preprocessing on training data
        high_freq_cats = {col: get_categories_high_freq(X[col]) for col in cat_cols}
        outlier_bounds = {col: calculate_upper_bound(X[col]) for col in continuous_cols}
        medians = {col: X[col].median() for col in processing_num_cols}
        
        artifacts = {
            'high_freq_cats': high_freq_cats,
            'outlier_bounds': outlier_bounds,
            'medians': medians,
            'cat_cols': cat_cols,
            'num_cols': processing_num_cols,
            'te_cols': te_cols,
            'ohe_cols': ohe_cols
        }
    else:
        high_freq_cats = artifacts['high_freq_cats']
        outlier_bounds = artifacts['outlier_bounds']
        medians = artifacts['medians']
        te_cols = artifacts.get('te_cols', [])
        ohe_cols = artifacts.get('ohe_cols', cat_cols)
    
    # 1. Categorical preprocessing
    if standardize_cats_flag:
        for col in cat_cols:
            X[col] = standardize_categorical_col(X[col], high_freq_cats[col])
            X[col] = X[col].fillna('other')
    else:
        # Even if not standardizing, we might want to fill NaNs or handle new categories
        # For simplicity, just fill NaNs with 'other' to avoid errors in encoding
        for col in cat_cols:
            X[col] = X[col].fillna('other')
    
    # 2. Numerical outliers
    if clean_outliers_flag:
        for col in continuous_cols:
            X[col] = clean_outliers(X[col], outlier_bounds[col])
    
    # 3. Fill missing values
    for col in processing_num_cols:
        X[col] = X[col].fillna(medians[col])
    
    # 4. Encoding
    encoded_dfs = []

    # Target Encoding
    if te_cols:
        if fit:
            if y is None:
                raise ValueError("Target variable 'y' is required for fitting TargetEncoder.")
            te = TargetEncoder(target_type='continuous', smooth="auto")
            te_data = pd.DataFrame(
                te.fit_transform(X[te_cols], y),
                columns=te.get_feature_names_out(te_cols),
                index=X.index
            )
            artifacts['target_encoder'] = te
        else:
            te = artifacts.get('target_encoder')
            if te:
                te_data = pd.DataFrame(
                    te.transform(X[te_cols]),
                    columns=te.get_feature_names_out(te_cols),
                    index=X.index
                )
            else:
                te_data = pd.DataFrame()
        encoded_dfs.append(te_data)

    # One-Hot Encoding
    if ohe_cols:
        if fit:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe_data = pd.DataFrame(
                ohe.fit_transform(X[ohe_cols]),
                columns=ohe.get_feature_names_out(ohe_cols),
                index=X.index
            )
            artifacts['encoder'] = ohe
        else:
            ohe = artifacts['encoder']
            ohe_data = pd.DataFrame(
                ohe.transform(X[ohe_cols]),
                columns=ohe.get_feature_names_out(ohe_cols),
                index=X.index
            )
        
        # Filter out 'other' columns generated by OHE
        cols_to_drop = [col for col in ohe_data.columns if col.endswith('_other')]
        ohe_data = ohe_data.drop(columns=cols_to_drop)
        
        encoded_dfs.append(ohe_data)
    
    X = X.drop(columns=cat_cols)
    if encoded_dfs:
        X = pd.concat([X] + encoded_dfs, axis=1)
    
    # 5. Normalize numerical features
    if normalize_flag:
        numerical_cols = list(processing_num_cols.keys())
        
        # Add target encoded columns to normalization
        if te_cols:
            numerical_cols.extend(te_cols)

        if fit:
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            artifacts['scaler'] = scaler
        else:
            X[numerical_cols] = artifacts['scaler'].transform(X[numerical_cols])
    
    return (X, artifacts) if fit else X

def sample_hyperparameters(param_distributions: dict, n_iter: int, seed: int) -> list[dict]:
    """
    Sample hyperparameters from distributions.
    
    Args:
        param_distributions (dict): Dictionary with parameter names as keys and
                                   distributions/lists as values.
        n_iter (int): Number of parameter combinations to sample.
        seed (int): Random seed for reproducibility.
        
    Returns:
        list[dict]: List of parameter dictionaries.
    """
    np.random.seed(seed)
    param_list = []
    
    for i in range(n_iter):
        params = {}
        for param_name, param_values in param_distributions.items():
            # Check if it's a scipy distribution (has .rvs method)
            if hasattr(param_values, 'rvs'):
                val = param_values.rvs(random_state=seed + i)
                # Convert numpy types to python types
                if isinstance(val, np.generic):
                    val = val.item()
                params[param_name] = val
            # Check if it's a list (discrete choices)
            elif isinstance(param_values, list):
                # Use random index to avoid issues with lists of tuples/objects
                idx = np.random.randint(0, len(param_values))
                val = param_values[idx]
                # Convert numpy types to python types
                if isinstance(val, np.generic):
                    val = val.item()
                params[param_name] = val
            else:
                # Assume it's a constant value (int, float, str, bool, etc.)
                params[param_name] = param_values
        
        param_list.append(params)
    
    return param_list

def cross_validate_with_tuning(X_raw: pd.DataFrame, 
                               y_raw: pd.Series, 
                               cat_cols_list: list[str], 
                               num_cols_dict: dict[str, str], 
                               model_config: dict, 
                               k: int = 3, 
                               seed: int = 42,
                               selected_features: list[str] | None = None,
                               log_target: bool = True,
                               verbose: bool = True,
                               preprocessing_params: dict | None = None) -> dict:
    """
    Perform k-fold cross-validation with manual hyperparameter search.
    Preprocessing is done within each fold to prevent data leakage.
    
    Supports tuning preprocessing parameters and log_target if they are included
    in model_config['param_distributions'].
    
    Args:
        X_raw (pd.DataFrame): Raw training features (not preprocessed).
        y_raw (pd.Series): Raw training target (not log-transformed).
        cat_cols_list (list[str]): List of categorical column names.
        num_cols_dict (dict[str, str]): Dictionary of numerical columns with types.
        model_config (dict): Configuration dictionary with keys:
            - 'model_class': sklearn model class (e.g., Ridge, Lasso, RandomForestRegressor).
            - 'param_distributions': dict of parameter distributions for sampling.
            - 'n_iter': number of parameter settings to sample (default: 20).
        k (int): Number of CV folds (default: 3).
        seed (int): Random seed for reproducibility.
        selected_features (list[str] | None): List of processed feature names to keep. 
                                              If None, all features are used.
        log_target (bool): Default value for log_target if not in param_distributions.
        verbose (bool): If True, print summary of results.
        preprocessing_params (dict | None): Default preprocessing flags if not in param_distributions.
        
    Returns:
        dict: Results with best_params, best_estimator, CV scores, and preprocessing artifacts.
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
    
    # Storage for results: param_idx -> {fold_idx: {'val_mae': ..., 'train_mae': ...}}
    param_results = {i: {'params': params, 'fold_scores': []} for i, params in enumerate(param_combinations)}
    
    # Store artifacts per fold to reconstruct fold_results later
    # Note: With varying preprocessing, artifacts depend on params AND fold.
    # We will store the artifacts for the BEST params later.
    
    if verbose:
        print(f"Starting {k}-Fold CV with {model_config['model_class'].__name__} ({n_iter} hyperparam combinations)...")
    
    # Perform manual CV
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_raw), 1):
        # Split data for this fold
        X_train_fold = X_raw.iloc[train_idx].copy()
        X_val_fold = X_raw.iloc[val_idx].copy()
        y_train_fold = y_raw.iloc[train_idx].copy()
        y_val_fold = y_raw.iloc[val_idx].copy()
        
        # Evaluate each parameter combination on this fold
        for i, params in enumerate(param_combinations):
            
            # 1. Determine Preprocessing Params
            current_preprocessing_params = preprocessing_params.copy() if preprocessing_params else {}
            
            # Set defaults if not provided
            defaults = {
                'clean_outliers_flag': True,
                'standardize_cats_flag': True,
                'normalize_flag': True
            }
            for key, val in defaults.items():
                if key not in current_preprocessing_params:
                    current_preprocessing_params[key] = val
                    
            # Override with sampled params
            pipeline_keys = ['clean_outliers_flag', 'standardize_cats_flag', 'normalize_flag']
            for key in pipeline_keys:
                if key in params:
                    current_preprocessing_params[key] = params[key]
            
            # 2. Determine Log Target
            current_log_target = log_target
            if 'log_target' in params:
                current_log_target = params['log_target']
                
            # 3. Determine Model Params
            model_params = {k: v for k, v in params.items() if k not in pipeline_keys and k != 'log_target' and k != 'feature_selection_k'}
            
            # Preprocess data for this fold AND this param combination
            X_train_processed, fold_artifacts = preprocess_data(
                X_train_fold, cat_cols_list, num_cols_dict, fit=True, y=y_train_fold, **current_preprocessing_params
            )
            X_val_processed = preprocess_data(
                X_val_fold, cat_cols_list, num_cols_dict, 
                artifacts=fold_artifacts, fit=False, **current_preprocessing_params
            )
            
            # Filter selected features if provided (static selection)
            if selected_features is not None:
                # Keep only features that exist in the processed data
                cols_to_keep = [c for c in selected_features if c in X_train_processed.columns]
                X_train_processed = X_train_processed[cols_to_keep]
                X_val_processed = X_val_processed[cols_to_keep]

            # Transform target if requested
            if current_log_target:
                y_train_target = np.log1p(y_train_fold)
            else:
                y_train_target = y_train_fold

            # 4. Dynamic Feature Selection (inside fold)
            current_k = params.get('feature_selection_k', None)
            if current_k is not None:
                # Ensure k is not larger than number of features
                k_val = min(int(current_k), X_train_processed.shape[1])
                
                selector = SelectKBest(f_regression, k=k_val)
                selector.fit(X_train_processed, y_train_target)
                
                # Get selected feature names
                selected_mask = selector.get_support()
                selected_cols = X_train_processed.columns[selected_mask].tolist()
                
                X_train_processed = X_train_processed[selected_cols]
                X_val_processed = X_val_processed[selected_cols]
            
            # Train model with these parameters
            model = model_config['model_class'](**model_params)
            model.fit(X_train_processed, y_train_target)
            
            # Predict on validation fold
            y_val_pred_raw = model.predict(X_val_processed)
            if current_log_target:
                y_val_pred = np.expm1(y_val_pred_raw)
            else:
                y_val_pred = y_val_pred_raw
                
            val_mae = mean_absolute_error(y_val_fold, y_val_pred)
            
            # Predict on train fold (for monitoring overfitting)
            y_train_pred_raw = model.predict(X_train_processed)
            if current_log_target:
                y_train_pred = np.expm1(y_train_pred_raw)
            else:
                y_train_pred = y_train_pred_raw
                
            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            
            # Store results
            param_results[i]['fold_scores'].append({
                'fold': fold_idx,
                'val_mae': val_mae,
                'train_mae': train_mae,
                # We don't store artifacts here to save memory, but we need them for the best model later.
                # Since we refit on all data, we don't strictly need fold artifacts for the return value,
                # except for the 'fold_results' display.
            })

    # Calculate stats for all parameter combinations
    summary_data = []
    for i, res in param_results.items():
        val_maes = [s['val_mae'] for s in res['fold_scores']]
        avg_val_mae = np.mean(val_maes)
        std_val_mae = np.std(val_maes)
        
        row = {
            'param_idx': i,
            'mean_val_mae': avg_val_mae,
            'std_val_mae': std_val_mae
        }
        # Flatten parameters into columns
        row.update(res['params'])
        
        summary_data.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by mean validation MAE
    summary_df = summary_df.sort_values('mean_val_mae')
    
    # Best is the first one
    best_param_idx = int(summary_df.iloc[0]['param_idx'])
    best_params_overall = param_combinations[best_param_idx]
    
    best_scores = param_results[best_param_idx]['fold_scores']
    
    # Construct fold_results for the best parameter set
    fold_results = []
    for score_data in best_scores:
        fold_results.append({
            'fold': score_data['fold'],
            'best_params': best_params_overall,
            'train_mae': score_data['train_mae'],
            'val_mae': score_data['val_mae'],
            'best_model': None, 
            'artifacts': None # We didn't save them per fold to save memory/complexity
        })
        
    # Calculate mean and std of CV scores for the best parameters
    cv_scores = [r['val_mae'] for r in fold_results]
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    best_fold_score = min(cv_scores)
    
    # Refit on all data using BEST parameters
    if verbose:
        print("\nRefitting best model on all available data...")
        
    # Extract best pipeline params
    best_pipeline_params = preprocessing_params.copy() if preprocessing_params else {}
    defaults = {
        'clean_outliers_flag': True,
        'standardize_cats_flag': True,
        'normalize_flag': True
    }
    for key, val in defaults.items():
        if key not in best_pipeline_params:
            best_pipeline_params[key] = val
            
    pipeline_keys = ['clean_outliers_flag', 'standardize_cats_flag', 'normalize_flag']
    for key in pipeline_keys:
        if key in best_params_overall:
            best_pipeline_params[key] = best_params_overall[key]
            
    best_log_target = log_target
    if 'log_target' in best_params_overall:
        best_log_target = best_params_overall['log_target']
        
    best_model_params = {k: v for k, v in best_params_overall.items() if k not in pipeline_keys and k != 'log_target' and k != 'feature_selection_k'}

    X_all_processed, final_artifacts = preprocess_data(
        X_raw, cat_cols_list, num_cols_dict, fit=True, y=y_raw, **best_pipeline_params
    )
    
    # Filter selected features if provided (static)
    if selected_features is not None:
        cols_to_keep = [c for c in selected_features if c in X_all_processed.columns]
        X_all_processed = X_all_processed[cols_to_keep]
        # Store selected features in artifacts for test time
        final_artifacts['selected_features'] = cols_to_keep
        
    final_artifacts['log_target'] = best_log_target
    # Store preprocessing params in artifacts for test time
    final_artifacts['preprocessing_params'] = best_pipeline_params
        
    if best_log_target:
        y_all_target = np.log1p(y_raw)
    else:
        y_all_target = y_raw

    # Apply Dynamic Feature Selection on all data if it was part of best params
    best_k = best_params_overall.get('feature_selection_k', None)
    if best_k is not None:
        k_val = min(int(best_k), X_all_processed.shape[1])
        
        selector = SelectKBest(f_regression, k=k_val)
        selector.fit(X_all_processed, y_all_target)
        
        selected_mask = selector.get_support()
        selected_cols = X_all_processed.columns[selected_mask].tolist()
        
        X_all_processed = X_all_processed[selected_cols]
        
        # Update artifacts so test data is filtered correctly
        final_artifacts['selected_features'] = selected_cols
    
    final_model = model_config['model_class'](**best_model_params)
    final_model.fit(X_all_processed, y_all_target)

    # Determine number of features selected
    n_features_selected = X_all_processed.shape[1]

    # Print summary table
    n_models_fitted = n_iter * k
    if verbose:
        print_cv_summary(
            fold_results,
            mean_cv_score,
            std_cv_score,
            best_fold_score,
            n_models_fitted,
            summary_df=summary_df,
            n_features_selected=n_features_selected
        )
    
    return {
        'best_params': best_params_overall,
        'best_estimator': final_model,
        'final_artifacts': final_artifacts,
        'mean_cv_score': mean_cv_score,
        'std_cv_score': std_cv_score,
        'best_fold_score': best_fold_score,
        'fold_results': fold_results,
        'results_summary': summary_df
    }

def print_cv_summary(fold_results: list[dict], 
                     mean_score: float, 
                     std_score: float, 
                     best_score: float, 
                     n_models_fitted: int, 
                     summary_df: pd.DataFrame | None = None,
                     n_features_selected: int | None = None) -> None:
    """
    Prints a summary of cross-validation results including top hyperparameters and fold performance.

    Args:
        fold_results (list[dict]): List of dictionaries containing results for each fold.
        mean_score (float): Mean validation score across all folds.
        std_score (float): Standard deviation of validation scores.
        best_score (float): Best single fold validation score.
        n_models_fitted (int): Total number of models trained during search.
        summary_df (pd.DataFrame | None): DataFrame containing hyperparameter search results.
        n_features_selected (int | None): Number of features selected by the final model.
    """
    print(f"Fitted {n_models_fitted} models in total.")
    
    if n_features_selected is not None:
        print(f"Features Selected: {n_features_selected}")

    if summary_df is not None:
        print(f"\nTop 5 Hyperparameter Combinations:")
        # Simple print of the top 5 rows, dropping the index column for cleaner output
        print(summary_df.drop(columns=['param_idx']).round(4).head(5).to_string(index=False))

    print(f"\nBest Model Performance (Across Folds):")
    print(f"{'Fold':<6} | {'Train MAE':<12} | {'Val MAE':<12}")
    print("-" * 36)
    
    for r in fold_results:
        print(f"{r['fold']:<6} | £{r['train_mae']:<10.2f} | £{r['val_mae']:<10.2f}")
    print("-" * 36)
    print(f"Mean CV MAE: £{mean_score:.2f} ± £{std_score:.2f}")
    print(f"Best Fold MAE: £{best_score:.2f}\n")

def preprocess_test_data(test_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Preprocess test data using artifacts from training.
    
    Args:
        test_df (pd.DataFrame): Raw test dataframe.
        artifacts (dict): Preprocessing artifacts from cross_validate_with_tuning.
        
    Returns:
        pd.DataFrame: Preprocessed test data ready for prediction.
    """
    # General cleaning
    test_cleaned = general_cleaning(test_df)
    
    # Extract preprocessing flags from artifacts if available, otherwise default to True
    preprocessing_params = artifacts.get('preprocessing_params', {})
    
    # Apply preprocessing using artifacts
    test_processed = preprocess_data(
        test_cleaned, 
        artifacts['cat_cols'], 
        artifacts['num_cols'], 
        artifacts=artifacts, 
        fit=False,
        **preprocessing_params
    )
    
    # Filter selected features if they were used during training
    if 'selected_features' in artifacts:
        cols_to_keep = [c for c in artifacts['selected_features'] if c in test_processed.columns]
        test_processed = test_processed[cols_to_keep]
    
    return test_processed

def get_feature_importance(fitted_model, X_train, model_class=None, plot=True, ax=None):
    """Extract and visualize feature importance from a fitted model.
    
    Extracts feature importance using the appropriate method based on model type:
    - Tree-based models: uses feature_importances_ attribute
    - Linear models (Ridge/Lasso): uses absolute coefficient values
    - MLPRegressor: uses L2 norm of first layer weights
    
    Args:
        fitted_model: An already-fitted model instance (Ridge, Lasso, RandomForestRegressor, 
                      ExtraTreesRegressor, GradientBoostingRegressor, or MLPRegressor).
        X_train (pd.DataFrame): Training features DataFrame used for column names in visualization.
        model_class (type, optional): Optional model class. If not provided, will be inferred from fitted_model.
                                      Useful for disambiguation if model type is ambiguous.
        plot (bool): Whether to display the feature importance plot. Default is True.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto, otherwise uses the current figure.
    
    Raises:
        ValueError: If model type is not supported or lacks extractable feature importance.
    
    Returns:
        pd.DataFrame: DataFrame containing feature importance values.
    """

    if model_class is None:
        model_class = type(fitted_model)
    
    results = []

    # Trees with feature_importances_
    if hasattr(fitted_model, 'feature_importances_'):
        importance = fitted_model.feature_importances_
        criterion = getattr(fitted_model, 'criterion', 'unknown')
        results.append((criterion, importance))

    # Ridge / Lasso (based on coef)
    elif hasattr(fitted_model, 'coef_'):
        coef = np.abs(fitted_model.coef_)
        if coef.ndim > 1:  
            coef = coef.mean(axis=0)
        results.append(("coef", coef))

    # MLP - Importance based on first layer weights
    elif model_class is MLPRegressor:
        first_layer_weights = fitted_model.coefs_[0]     # shape = (n_features, n_hidden)
        importance = np.linalg.norm(first_layer_weights, axis=1)
        results.append(("mlp_weights", importance))

    else:
        raise ValueError(f"Model {model_class.__name__} not supported or has no extractable importance.")

    df_list = []
    for label, values in results:
        df_list.append(pd.DataFrame({
            "Feature": X_train.columns,
            "Value": values,
            "Method": label
        }))

    tidy = pd.concat(df_list)
    
    # Filter out "other" columns generated by OHE
    tidy = tidy[~tidy['Feature'].str.endswith('_other')]
    
    tidy.sort_values("Value", ascending=False, inplace=True)

    if plot:
        if ax is None:
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
        
        sns.barplot(data=tidy, y="Feature", x="Value", hue="Method", ax=ax)
        ax.set_title(f"Feature Importance — {model_class.__name__}")
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    return tidy
