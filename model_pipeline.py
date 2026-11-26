"""
model_pipeline.py
=================
Modular version of the notebook "Projet_ML_25_variables25.ipynb".

Predicts hospital readmission of diabetic patients
(dataset: Diabetes 130-US Hospitals 1999-2008).

Pipeline steps:
    prepare_data()    -> load, clean, feature engineering, train/test split
    select_features() -> ANOVA (filter) + Lasso (embedded), keep the common variables
    train_model()     -> train the model (optionally tuned with RandomizedSearchCV)
    evaluate_model()  -> metrics on the test set
    save_model()      -> save the trained pipeline with joblib
    load_model()      -> load a saved pipeline
"""

import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "readmis"

# Patients who died or were sent to a hospice cannot be readmitted
NON_READMISSIBLE_CODES = [11, 13, 14, 19, 20, 21]

MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

XGB_PARAM_GRID = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "gamma": [0, 0.5, 1, 2],
    "reg_lambda": [1, 5, 10],
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _diagnosis_group(code):
    """Map an ICD-9 code to one of 9 disease families."""
    if code == "Unknown":
        return "Unknown"
    if code.startswith(("V", "E")):
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _build_preprocessor(num_features, cat_features):
    """StandardScaler for numeric columns, One-Hot Encoding for categorical ones."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        verbose_feature_names_out=False,
    )


def _default_xgb():
    """XGBoost with the notebook's starting settings."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _select_columns(X, indices):
    """Keep only the selected encoded columns (used inside the saved Pipeline)."""
    return np.take(X, indices, axis=1)


# ----------------------------------------------------------------------------
# 1. Data preparation
# ----------------------------------------------------------------------------
def prepare_data(data_path="diabetic_data.csv", test_size_folds=5):
    """
    Load and preprocess the data, then split it into train / test.

    Steps (same as notebook parts 1, 3 and 4.1):
      - binary target `readmis` (1 = readmitted, whatever the delay)
      - remove duplicates and non-readmissible patients
      - handle missing values ('?' -> NaN -> imputed / dropped)
      - drop useless columns
      - feature engineering (age_num, diagnosis families, total_visites, ...)
      - stratified split grouped by patient (80 % train / 20 % test)

    Args:
        data_path: path to diabetic_data.csv
        test_size_folds: 5 -> 1/5 = 20 % of the patients go to the test set

    Returns:
        dict with X_train, X_test, y_train, y_test, groups_train
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[prepare] Loaded {data_path}: {df.shape}")

    # Target
    df[TARGET] = (df["readmitted"] != "NO").astype(int)

    # Duplicates and non-readmissible patients
    df = df.drop_duplicates().copy()
    df = df[~df["discharge_disposition_id"].isin(NON_READMISSIBLE_CODES)].copy()

    # Missing values
    df = df.replace("?", np.nan)
    df = df.drop(columns=["weight"])
    for col in ["max_glu_serum", "A1Cresult"]:
        df[col] = df[col].fillna("None")
    for col in ["medical_specialty", "payer_code", "race", "diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].fillna("Unknown")

    # Useless columns (patient_nbr kept apart for the grouped split)
    groups = df["patient_nbr"].copy()
    df = df.drop(columns=["encounter_id", "patient_nbr", "examide", "citoglipton", "readmitted"])

    # Feature engineering
    age_mapping = {f"[{10 * i}-{10 * (i + 1)})": 10 * i + 5 for i in range(10)}
    df["age_num"] = df["age"].map(age_mapping)
    df = df.drop(columns=["age"])

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].astype(str).apply(_diagnosis_group)

    df["total_visites"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["nb_changements_med"] = df[MEDICATIONS].isin(["Up", "Down"]).sum(axis=1)

    known = df["medical_specialty"] != "Unknown"
    top_specialties = df.loc[known, "medical_specialty"].value_counts().index[:10]
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_specialties) | ~known, "Other"
    )

    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    print(f"[prepare] After cleaning: {df.shape}, readmitted = {df[TARGET].mean():.1%}")

    # Train / test split: stratified and grouped by patient
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    sgkf = StratifiedGroupKFold(n_splits=test_size_folds, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    data = {
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "groups_train": groups.iloc[train_idx].copy(),
    }
    print(f"[prepare] X_train: {data['X_train'].shape} | X_test: {data['X_test'].shape}")
    return data


# ----------------------------------------------------------------------------
# 2. Feature selection
# ----------------------------------------------------------------------------
def select_features(
    X_train,
    y_train,
    groups_train,
    k_anova=30,
    lasso_c_values=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
):
    """
    Select the encoded variables kept by BOTH the ANOVA filter and the Lasso
    (notebook part 5). Uses the training set only.

    Returns:
        list of selected encoded feature names (e.g. 'number_inpatient',
        'admission_source_id_7', ...)
    """
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = _build_preprocessor(num_features, cat_features)
    X_train_enc = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    print(f"[select] {X_train_enc.shape[1]} columns after encoding")

    # Filter method: ANOVA
    anova = SelectKBest(score_func=f_classif, k=k_anova).fit(X_train_enc, y_train)
    features_anova = X_train_enc.columns[anova.get_support()].tolist()

    # Embedded method: Lasso (L1 logistic regression), C chosen by grouped CV
    cv_lasso = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE).split(
            X_train_enc, y_train, groups_train
        )
    )
    results = []
    for c in lasso_c_values:
        lasso = LogisticRegression(
            penalty="l1", solver="liblinear", C=c, max_iter=1000, random_state=RANDOM_STATE
        )
        auc = cross_validate(lasso, X_train_enc, y_train, cv=cv_lasso, scoring="roc_auc")[
            "test_score"
        ].mean()
        results.append((c, auc))
        print(f"[select] Lasso C={c:<7} AUC(CV)={auc:.4f}")

    # Smallest C whose AUC is within 0.002 of the best one
    best_auc = max(auc for _, auc in results)
    c_lasso = min(c for c, auc in results if auc >= best_auc - 0.002)

    lasso_final = LogisticRegression(
        penalty="l1", solver="liblinear", C=c_lasso, max_iter=1000, random_state=RANDOM_STATE
    )
    lasso_final.fit(X_train_enc, y_train)
    coefs = pd.Series(lasso_final.coef_[0], index=X_train_enc.columns)
    features_lasso = coefs[coefs != 0].index.tolist()

    features_sel = sorted(set(features_anova) & set(features_lasso))
    print(
        f"[select] C={c_lasso} | ANOVA: {len(features_anova)} | Lasso: {len(features_lasso)} "
        f"| common: {len(features_sel)}"
    )
    return features_sel


# ----------------------------------------------------------------------------
# 3. Training
# ----------------------------------------------------------------------------
def train_model(X_train, y_train, features_sel, groups_train=None, optimize=False, n_iter=30):
    """
    Train an XGBoost model on the selected variables and wrap it in a Pipeline
    that takes the ORIGINAL (raw) columns as input (notebook parts 6, 7.3, 7.7).

    Args:
        X_train, y_train: training data (raw columns, from prepare_data)
        features_sel: encoded variables to keep (from select_features)
        groups_train: patient ids, needed only when optimize=True
        optimize: tune hyperparameters with RandomizedSearchCV (slow: ~10 min)
        n_iter: number of combinations tested when optimize=True

    Returns:
        (pipeline, columns): the fitted Pipeline and the list of raw columns it needs
    """
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Raw column behind each encoded variable (e.g. admission_source_id_7 -> admission_source_id)
    full_pre = _build_preprocessor(num_features, cat_features).fit(X_train)
    encoder = full_pre.named_transformers_["cat"]
    origin = {f: f for f in num_features}
    for col, cats in zip(encoder.feature_names_in_, encoder.categories_):
        for cat in cats:
            origin[f"{col}_{cat}"] = col
    needed = {origin[f] for f in features_sel}

    columns = [c for c in X_train.columns if c in needed]
    num_used = [c for c in num_features if c in needed]
    cat_used = [c for c in cat_features if c in needed]

    preprocessor = _build_preprocessor(num_used, cat_used).fit(X_train[columns])
    out_names = list(preprocessor.get_feature_names_out())
    indices = [out_names.index(f) for f in features_sel]

    model = _default_xgb()

    if optimize:
        if groups_train is None:
            raise ValueError("groups_train is required when optimize=True")
        X_sel = _select_columns(preprocessor.transform(X_train[columns]), indices)
        cv = list(
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
                X_sel, y_train, groups_train
            )
        )
        search = RandomizedSearchCV(
            model,
            XGB_PARAM_GRID,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        start = time.time()
        search.fit(X_sel, y_train)
        print(
            f"[train] RandomizedSearchCV done in {time.time() - start:.0f}s, "
            f"best AUC(CV)={search.best_score_:.4f}"
        )
        print(f"[train] Best params: {search.best_params_}")
        model = search.best_estimator_  # refitted below inside the Pipeline

    pipeline = Pipeline(
        [
            ("pretraitement", preprocessor),
            ("selection", FunctionTransformer(_select_columns, kw_args={"indices": indices})),
            ("modele", model),
        ]
    )

    start = time.time()
    pipeline.fit(X_train[columns], y_train)
    print(
        f"[train] Model trained in {time.time() - start:.1f}s on {len(features_sel)} variables "
        f"({len(columns)} raw columns)"
    )
    return pipeline, columns


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------
def evaluate_model(pipeline, columns, X_test, y_test):
    """
    Evaluate the model on the test set (notebook part 7.4).

    Returns:
        dict of metrics: accuracy, precision, recall, f1, roc_auc
    """
    X = X_test[columns]
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("[evaluate] Test set results:")
    for name, value in metrics.items():
        print(f"    {name:<10}: {value:.4f}")
    print("\n[evaluate] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n[evaluate] Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Non readmis", "Readmis"]))
    return metrics


# ----------------------------------------------------------------------------
# 5. Save / load
# ----------------------------------------------------------------------------
def save_model(pipeline, columns, path="models/modele_readmission_25.pkl"):
    """Save the trained pipeline and the list of raw columns it expects (joblib)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "colonnes": columns}, path)
    print(f"[save] Model saved to {path}")
    return path


def load_model(path="models/modele_readmission_25.pkl"):
    """Load a pipeline saved with save_model(). Returns (pipeline, columns)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path} (run the 'train' step first)")
    saved = joblib.load(path)
    print(f"[load] Model loaded from {path}")
    return saved["pipeline"], saved["colonnes"]


# ----------------------------------------------------------------------------
# 6. Prediction
# ----------------------------------------------------------------------------
def predict(pipeline, columns, X):
    """
    Predict readmission for new patients.

    Args:
        pipeline, columns: as returned by load_model()
        X: DataFrame with (at least) the raw columns the model needs,
           already prepared like the output of prepare_data()

    Returns:
        DataFrame with 'prediction' (0/1) and 'proba_readmis' (probability of class 1)
    """
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")
    X = X[columns]
    result = pd.DataFrame(
        {
            "prediction": pipeline.predict(X),
            "proba_readmis": pipeline.predict_proba(X)[:, 1],
        },
        index=X.index,
    )
    rate = result["prediction"].mean()
    print(f"[predict] {len(result)} predictions, {rate:.1%} predicted readmitted")
    return result
