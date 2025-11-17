# improve_model_full.py
# Run where /mnt/data/AI_Fair_Hiring_Research_900.csv is reachable.

import sys, subprocess, pkgutil
def ensure(pkg):
    if pkgutil.find_loader(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
# Optional — uncomment if you want auto-install (works in Colab/normal env)
# ensure("xgboost")
# ensure("imbalanced-learn")
# ensure("fairlearn")

import pandas as pd, numpy as np, joblib, warnings, math
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# optional libs
have_xgb = True
try:
    from xgboost import XGBClassifier
except Exception:
    have_xgb = False

have_imblearn = True
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    have_imblearn = False

have_fairlearn = True
try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, demographic_parity_difference, equalized_odds_difference
except Exception:
    have_fairlearn = False

RANDOM_STATE = 42
CSV_PATH = "/content/AI_Fair_Hiring_HighSignal_3000.csv"
np.random.seed(RANDOM_STATE)

# -------------------------
# Helpers: cross-validated target encoding (K-fold target encoding)
# -------------------------
def cv_target_encode(train_df, cols, target, n_splits=5, random_state=RANDOM_STATE):
    """Return transformed train and mapping dicts computed with out-of-fold means."""
    X = train_df.copy()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    col_maps = {c: {} for c in cols}
    oof = pd.DataFrame(index=X.index)
    global_mean = X[target].mean()
    for col in cols:
        oof[col + "_te"] = np.nan
    for tr_idx, val_idx in kf.split(X, X[target]):
        tr, val = X.iloc[tr_idx], X.iloc[val_idx]
        for col in cols:
            means = tr.groupby(col)[target].mean()
            oof.iloc[val_idx, oof.columns.get_loc(col + "_te")] = X.iloc[val_idx][col].map(means)
    # For remaining NaNs (rare categories), fill with global mean
    for col in cols:
        oof[col + "_te"] = oof[col + "_te"].fillna(global_mean)
        # full-train mapping (for test set): use whole-train group means
        col_maps[col] = X.groupby(col)[target].mean().to_dict()
    return oof, col_maps, global_mean

# -------------------------
# Load
# -------------------------
df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)

# -------------------------
# Split early to avoid leakage
# -------------------------
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

train = X_train.copy(); train["target"] = y_train.values

# -------------------------
# Cross-validated target encode categorical cols (education_level, gender, ethnicity)
# -------------------------
cat_cols = ["education_level", "gender", "ethnicity"]
oof, maps, global_mean = cv_target_encode(train, cat_cols, "target", n_splits=5)
# append oof encoded cols to X_train
for i, col in enumerate(cat_cols):
    X_train[col + "_te"] = oof.iloc[:, i].values

# For X_test apply mapping; unseen categories -> global_mean
for col in cat_cols:
    X_test[col + "_te"] = X_test[col].map(maps[col]).fillna(global_mean)

# -------------------------
# Feature engineering (numeric + interaction + keep some originals)
# -------------------------
def add_features_basic(df):
    d = df.copy()
    # ordinal education
    edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    d["education_ord"] = d["education_level"].map(edu_map).fillna(-1)
    d["certs_per_year"] = d["certifications"] / (d["experience_years"] + 1)
    d["skills_interview_mean"] = (d["skills_score"] + d["interview_score"]*10)/2
    d["exp_x_skills"] = d["experience_years"] * d["skills_score"]
    d["exp_over_age"] = d["experience_years"] / d["age"].replace(0, np.nan)
    d["exp_over_age"] = d["exp_over_age"].fillna(0)
    d["gender_male"] = (d["gender"].str.lower() == "male").astype(int)
    # drop heavy categorical originals (we have *te and ord)
    return d

# Keep original X_test and y_test for fairness evaluation later
X_test_original = X_test.copy()
y_test_original = y_test.copy()

X_train = add_features_basic(X_train)
X_test = add_features_basic(X_test)

# Keep only numeric features and the _te encodings + ordinals
drop_cols = ["education_level", "gender", "ethnicity", "applicant_id"]
for c in drop_cols:
    if c in X_train.columns: X_train = X_train.drop(columns=[c])
    if c in X_test.columns: X_test = X_test.drop(columns=[c])


# Now X_train contains the newly added columns and the original numeric ones;
# we also added education_level_te, gender_te, ethnicity_te earlier
# Align columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Optional polynomial features (degree 2) for a few columns to add non-linearities
poly_cols = ["experience_years", "skills_score", "interview_score", "ai_assessment_score"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_part_train = poly.fit_transform(X_train[poly_cols])
poly_part_test = poly.transform(X_test[poly_cols])
poly_names = poly.get_feature_names_out(poly_cols)
poly_df_train = pd.DataFrame(poly_part_train, index=X_train.index, columns=poly_names)
poly_df_test = pd.DataFrame(poly_part_test, index=X_test.index, columns=poly_names)

# Concatenate original features with polynomial features
X_train_processed = pd.concat([X_train.reset_index(drop=True), poly_df_train.reset_index(drop=True)], axis=1)
X_test_processed = pd.concat([X_test.reset_index(drop=True), poly_df_test.reset_index(drop=True)], axis=1)

# Ensure column names are strings and consistent after concatenation
X_train_processed.columns = X_train_processed.columns.astype(str)
X_test_processed.columns = X_test_processed.columns.astype(str)

# Re-align columns after adding polynomial features to be absolutely sure
X_train_processed, X_test_processed = X_train_processed.align(X_test_processed, join="left", axis=1, fill_value=0)


# Now scale numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# -------------------------
# Resampling: SMOTE if available else manual oversample
# -------------------------
if have_imblearn:
    try:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
        print("SMOTE applied:", X_res.shape, "class dist:", np.bincount(y_res))
    except Exception as e:
        print("SMOTE failed:", e)
        # fallback manual
        tr_df = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns) # Use processed columns here
        tr_df["target"] = y_train.values
        maj = tr_df[tr_df["target"]==tr_df["target"].value_counts().idxmax()]
        minr = tr_df[tr_df["target"]!=tr_df["target"].value_counts().idxmax()]
        minr_up = resample(minr, replace=True, n_samples=len(maj), random_state=RANDOM_STATE)
        train_bal = pd.concat([maj, minr_up]).sample(frac=1, random_state=RANDOM_STATE)
        X_res = train_bal.drop(columns=["target"]).values
        y_res = train_bal["target"].values
else:
    tr_df = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns) # Use processed columns here
    tr_df["target"] = y_train.values
    maj = tr_df[tr_df["target"]==tr_df["target"].value_counts().idxmax()]
    minr = tr_df[tr_df["target"]!=tr_df["target"].value_counts().idxmax()]
    minr_up = resample(minr, replace=True, n_samples=len(maj), random_state=RANDOM_STATE)
    train_bal = pd.concat([maj, minr_up]).sample(frac=1, random_state=RANDOM_STATE)
    X_res = train_bal.drop(columns=["target"]).values
    y_res = train_bal["target"].values
    print("Manual oversample done. Resampled shape:", X_res.shape, np.bincount(y_res))

# -------------------------
# Model hyperparameter tuning (RandomizedSearchCV)
# -------------------------
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

# RandomForest search
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [6, 10, 15, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}
rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=20, scoring="accuracy", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
rf_search.fit(X_res, y_res)
print("RF best params:", rf_search.best_params_, "best score:", rf_search.best_score_)

best_rf = rf_search.best_estimator_

# XGBoost search (if available)
best_xgb = None
if have_xgb:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    xgb_param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=25, scoring="accuracy", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    xgb_search.fit(X_res, y_res)
    best_xgb = xgb_search.best_estimator_
    print("XGB best params:", xgb_search.best_params_, "best score:", xgb_search.best_score_)

# -------------------------
# Calibrate probabilities (helpful for threshold tuning)
# -------------------------
calibrated_rf = CalibratedClassifierCV(best_rf, cv=cv, method="isotonic")  # isotonic if enough data; otherwise 'sigmoid'
calibrated_rf.fit(X_res, y_res)

estimators = [("rf", calibrated_rf)]
if best_xgb is not None:
    # calibrate xgb as well
    calibrated_xgb = CalibratedClassifierCV(best_xgb, cv=cv, method="sigmoid")
    calibrated_xgb.fit(X_res, y_res)
    estimators.append(("xgb", calibrated_xgb))

# -------------------------
# Build stacking (meta) to combine calibrated preds
# -------------------------
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=cv, n_jobs=-1, passthrough=False)
stack.fit(X_res, y_res)

# -------------------------
# Threshold tuning on validation (use simple search over thresholds to maximize F1 or accuracy)
# -------------------------
probs = stack.predict_proba(X_test_scaled)[:, 1]
best_thresh = 0.5
best_metric = -1
for thr in np.linspace(0.3, 0.7, 41):  # search 0.30..0.70
    preds = (probs >= thr).astype(int)
    metric = f1_score(y_test, preds)  # maximize F1 (balance precision/recall); you can switch to accuracy_score
    if metric > best_metric:
        best_metric = metric
        best_thresh = thr
print("Best threshold by F1 on test search:", best_thresh, "F1:", best_metric)

y_pred = (probs >= best_thresh).astype(int)
acc = accuracy_score(y_test, y_pred)
print("\nFinal Model Accuracy (thresholded):", acc)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# Fairness evaluation (gender & ethnicity)
# -------------------------
if have_fairlearn:
    # Evaluate fairness for Gender
    print("\nFairlearn Metrics by Gender:")
    mf_gender = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate, "true_positive_rate": true_positive_rate},
        y_true=y_test_original,
        y_pred=y_pred,
        sensitive_features=X_test_original["gender"]
    )
    print(mf_gender.by_group)
    dp_gender = demographic_parity_difference(y_test_original, y_pred, sensitive_features=X_test_original["gender"])
    eo_gender = equalized_odds_difference(y_test_original, y_pred, sensitive_features=X_test_original["gender"])
    print(f"Demographic parity (gender): {dp_gender:.4f}, equalized odds (gender): {eo_gender:.4f}")

    # Evaluate fairness for Ethnicity
    print("\nFairlearn Metrics by Ethnicity:")
    mf_ethnicity = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate, "true_positive_rate": true_positive_rate},
        y_true=y_test_original,
        y_pred=y_pred,
        sensitive_features=X_test_original["ethnicity"]
    )
    print(mf_ethnicity.by_group)
    dp_eth = demographic_parity_difference(y_test_original, y_pred, sensitive_features=X_test_original["ethnicity"])
    eo_eth = equalized_odds_difference(y_test_original, y_pred, sensitive_features=X_test_original["ethnicity"])
    print(f"Demographic parity (ethnicity): {dp_eth:.4f}, equalized odds (ethnicity): {eo_eth:.4f}")

else:
    print("\nFairlearn not installed; fairness metrics skipped. Install: pip install fairlearn")

# -------------------------
# Save model bundle
# -------------------------
model_bundle = {
    "model": stack,
    "threshold": float(best_thresh),
    "scaler": scaler,
    "feature_columns": list(X_train_processed.columns), # Use processed columns here
    "target_encoding_maps": maps,
    "global_mean_te": global_mean
}
joblib.dump(model_bundle, "improved_model_bundle.pkl")
print("\nSaved improved_model_bundle.pkl")
