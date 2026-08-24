import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# =====================================================================
# 1. AUTOMATED DATA VALIDATION FUNCTION
# =====================================================================
def validate_dataset(df, feature_cols, target_col):
    """
    Scans the dataset for missing cells, invalid data types, or out-of-bounds 
    network metrics before passing data to XGBoost.
    """
    print(" Running Pre-Training Data Validation...")
    errors = []
    
    # Check 1: Missing or blank cells (NaNs)
    missing_count = df[feature_cols + [target_col]].isnull().sum().sum()
    if missing_count > 0:
        errors.append(f"Found {missing_count} missing/blank cells in feature or target columns!")
        
    # Check 2: Non-numeric data types in feature set
    non_numeric = df[feature_cols].select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        errors.append(f"Non-numeric values found in feature columns: {list(non_numeric)}")
        
    # Check 3: Current BBP range bounds (must be between 0.0 and 1.0)
    if "current_BBP" in df.columns:
        invalid_bbp = df[(df["current_BBP"] < 0) | (df["current_BBP"] > 1.0)]
        if len(invalid_bbp) > 0:
            errors.append(f"Found {len(invalid_bbp)} rows with invalid current_BBP values outside [0.0, 1.0]!")

    # Check 4: Link technology flags (must be valid non-negative integers/counts)
    link_cols = [c for c in feature_cols if c.startswith("link_") and ("_SC_" in c or "_MC_" in c)]
    if link_cols:
        invalid_flags = (df[link_cols] < 0).any().any()
        if invalid_flags:
            errors.append("Some link technology features contain negative values!")

    # Check 5: Target column values (must be 0 or 1)
    
    if not df[target_col].isin([0, 1]).all():
        errors.append(f"Target column '{target_col}' contains values other than 0 or 1!")

    # Summary
    if errors:
        print("DATA VALIDATION FAILED:")
        for err in errors:
            print(f"   - {err}")
        return False
    else:
        print("✅ DATA VALIDATION PASSED: All inputs are clean and ready for training!\n")
        return True


# =====================================================================
# 2. DATA LOADING & COLUMN IDENTIFICATION
# =====================================================================
# Gets the directory where train_gatekeeper.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Points to the Test folder inside ML: .../ml/Test/sample_test.csv
file_path = os.path.join(SCRIPT_DIR, "XGBOOST", "Increment_dataset.csv")

print(f"Loading data from: {file_path}")
try:
    df = pd.read_csv(file_path) #, sheet_name=sheet_name)
    print(f"Successfully loaded dataset: {df.shape[0]} Rows x {df.shape[1]} Columns\n")
except Exception as e:
    print(f" Error loading Excel file: {e}")
    sys.exit(1)

# Metadata columns MUST be excluded from feature set (X)
metadata_cols = ["seed", "split", "algorithm", "cycle_number", "current_day"]
target_col = "upgrade_needed"

# Dynamically select all feature columns (609 features)
feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]

print(f"Metadata Columns (Excluded from training): {metadata_cols}")
print(f"Target Column (Y): '{target_col}'")
print(f"Input Features Count (X): {len(feature_cols)} features\n")


# =====================================================================
# 3. RUN VALIDATION CHECK
# =====================================================================
if not validate_dataset(df, feature_cols, target_col):
    print("Stopping pipeline due to validation errors. Fix the input file and re-run.")
    sys.exit(1)


# =====================================================================
# 4. TRAIN / TEST SPLITTING BY SEED
# =====================================================================
# Train: Seeds 1 to 15 | Test: Seeds 16 to 20
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

print(f"Train Set Shape: {X_train.shape[0]} rows x {X_train.shape[1]} features (Seeds 0–14)")
print(f"Test Set Shape:  {X_test.shape[0]} rows x {X_test.shape[1]} features (Seeds 15–19)\n")

# Compute scale_pos_weight for handling class imbalance (0s vs 1s)
num_zeros = (y_train == 0).sum()
num_ones = (y_train == 1).sum()
pos_ratio = num_zeros / num_ones if num_ones > 0 else 1.0
print(f"Class Distribution in Train: {num_zeros} No-Upgrades (0), {num_ones} Upgrades (1)")
print(f"Calculated scale_pos_weight: {pos_ratio:.2f}\n")


# =====================================================================
# 5. INITIALIZE & TRAIN XGBOOST MODEL
# =====================================================================
model = xgb.XGBClassifier(
    n_estimators=200,            # Max number of decision trees
    max_depth=5,                 # Depth of each tree (prevents overfitting)
    learning_rate=0.03,          # Step size shrinkage
    subsample=0.8,               # Sample 80% of rows per tree
    colsample_bytree=0.8,        # Sample 80% of features per tree
    scale_pos_weight=pos_ratio,  # Adjusts weights for imbalanced upgrade labels
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20     # Stops early if test loss stops improving
)

print("Starting XGBoost Model Training...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)
print("Training completed!\n")


# =====================================================================
# 6. EVALUATE PREDICTIONS ON HELD-OUT TEST SET
# =====================================================================
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("                    EVALUATION RESULTS                       ")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Upgrade Needed (0)", "Upgrade Needed (1)"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"[[ True Negatives (0s): {cm[0][0]} | False Positives (0s as 1): {cm[0][1]} ]")
print(f" [ False Negatives (1s as 0): {cm[1][0]} | True Positives (1s): {cm[1][1]} ]]\n")


# =====================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =====================================================================
importances = pd.Series(model.feature_importances_, index=feature_cols)
top_15_features = importances.nlargest(15)

print("=" * 60)
print("          TOP 15 MOST IMPORTANT NETWORK FEATURES             ")
print("=" * 60)
for rank, (feat_name, score) in enumerate(top_15_features.items(), 1):
    print(f"{rank:2d}. {feat_name:<35} | Importance Score: {score:.5f}")


# =====================================================================
# 8. SAVE MODEL FOR SIMULATOR INTEGRATION
# =====================================================================
output_model_name = "xgboost_gatekeeper_model.json"
model.save_model(output_model_name)
print("\n" + "=" * 60)
print(f"✅ Model exported successfully as '{output_model_name}'!")
print("=" * 60)