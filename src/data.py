import pandas as pd
import numpy as np

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
# This module handles loading and preprocessing of insurance claim data.
# The target variable is ClaimFrequency (number of claims per unit exposure).

def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess insurance claim data for modeling.
    
    This function performs the following steps:
    1. Load train and test CSV files
    2. Create target variable (ClaimFrequency = ClaimNb / Exposure)
    3. Filter out very small exposures
    4. Clip extreme claim frequencies to reduce outlier impact
    5. Create feature matrices with numerical and categorical features
    6. One-hot encode categorical variables
    7. Standardize features for neural network training
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
    
    Returns:
        Tuple containing:
        - df_train: Preprocessed training DataFrame
        - df_test: Preprocessed test DataFrame
        - X_train: Raw training features (numpy array)
        - X_test: Raw test features (numpy array)
        - y_train: Training targets (ClaimFrequency)
        - y_test: Test targets (ClaimFrequency)
        - X_train_std: Standardized training features (for neural networks)
        - X_test_std: Standardized test features (for neural networks)
    """
    # -----------------------------
    # 1. Load data
    # -----------------------------
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # -----------------------------
    # 2. Target: ClaimFrequency
    # -----------------------------
    # ClaimFrequency represents the number of claims per unit of exposure time
    # This normalizes claims by how long the policy was active
    for df in [df_train, df_test]:
        df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]

    # -----------------------------
    # 3. Filter tiny exposures
    # -----------------------------
    # Remove records with very small exposure times (< 0.01)
    # These can create unstable/unreliable claim frequency values
    min_exposure = 0.01
    df_train = df_train[df_train["Exposure"] >= min_exposure].copy()
    df_test  = df_test[df_test["Exposure"]  >= min_exposure].copy()

    # -----------------------------
    # 4. Clip extreme ClaimFrequency
    # -----------------------------
    # Cap claim frequency at the 99.5th percentile of training data
    # This reduces the impact of extreme outliers on model training
    cap = df_train["ClaimFrequency"].quantile(0.995)
    df_train["ClaimFrequency"] = df_train["ClaimFrequency"].clip(upper=cap)
    df_test["ClaimFrequency"]  = df_test["ClaimFrequency"].clip(upper=cap)

    # -----------------------------
    # 5. Features
    # -----------------------------
    # Numerical features: continuous variables
    num_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Exposure"]
    # Categorical features: discrete categories (e.g., geographic area)
    cat_cols = ["Area"]

    # One-hot encode categorical variables
    # This converts categorical values into binary columns (0 or 1)
    # drop_first=False keeps all categories (no reference category dropped)
    train_dummies = pd.get_dummies(df_train[cat_cols], drop_first=False)
    test_dummies  = pd.get_dummies(df_test[cat_cols],  drop_first=False)

    # Align columns in train/test to ensure they have the same features
    # If test set has categories not in train, they're filled with 0
    # This ensures both datasets have identical feature columns
    train_dummies, test_dummies = train_dummies.align(test_dummies, join="left", axis=1, fill_value=0)

    # Full design matrices as DataFrames
    # Combine numerical features with one-hot encoded categorical features
    X_train_df = pd.concat([df_train[num_cols], train_dummies], axis=1)
    X_test_df  = pd.concat([df_test[num_cols],  test_dummies],  axis=1)

    # Targets (what we're trying to predict)
    y_train = df_train["ClaimFrequency"].values.astype(np.float64)
    y_test  = df_test["ClaimFrequency"].values.astype(np.float64)

    # Numpy design matrices (raw features for decision tree)
    # Decision trees don't require standardization
    X_train = X_train_df.values.astype(np.float64)
    X_test  = X_test_df.values.astype(np.float64)

    # -----------------------------
    # 6. Standardization for NN
    # -----------------------------
    # Neural networks perform better with standardized inputs
    # Standardization: (X - mean) / std
    # This centers data around 0 with unit variance
    
    # Calculate mean and std from training data only (avoid data leakage)
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std  = X_train.std(axis=0, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero

    # Apply same standardization to both train and test
    # Important: use training statistics for both to avoid data leakage
    X_train_std = (X_train - X_mean) / X_std
    X_test_std  = (X_test  - X_mean) / X_std

    return df_train, df_test, X_train, X_test, y_train, y_test, X_train_std, X_test_std

