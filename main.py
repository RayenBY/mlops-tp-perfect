"""
main.py
=======
Runs the steps of model_pipeline.py from the command line, one step at a time.

Usage (inside the venv), in this order:
    python main.py prepare           # prepare_data()    -> artifacts/prepared_data.joblib
    python main.py select            # select_features() -> artifacts/selected_features.joblib
    python main.py train             # train_model()     -> artifacts/trained_model.joblib
    python main.py train --optimize  # same, with hyperparameter tuning (slow)
    python main.py save              # save_model()      -> models/modele_readmission_25.pkl
    python main.py load              # load_model(): check the saved model can be loaded
    python main.py evaluate          # load_model() + evaluate_model() on the test set
    python main.py predict           # load_model() + predict() on test patients
    python main.py all               # every step above, in order

Options:
    --data       path to diabetic_data.csv     (default: data/diabetic_data.csv)
    --model      path of the saved model       (default: models/modele_readmission_25.pkl)
    --n-samples  number of test patients to predict (default: 10)
"""

import argparse
import os

import joblib

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    select_features,
    train_model,
)

PREPARED_PATH = "artifacts/prepared_data.joblib"
FEATURES_PATH = "artifacts/selected_features.joblib"
TRAINED_PATH = "artifacts/trained_model.joblib"  # model kept between 'train' and 'save'
PREDICTIONS_PATH = "artifacts/predictions.csv"


def _load_artifact(path, step):
    """Load the output of a previous step, or explain which command to run first."""
    if not os.path.exists(path):
        raise SystemExit(f"[main] {path} not found. Run 'python main.py {step}' first.")
    print(f"[main] Loading {path}")
    return joblib.load(path)


def run_prepare(args):
    """Step 1: load, clean and split the data."""
    data = prepare_data(args.data)
    os.makedirs(os.path.dirname(PREPARED_PATH), exist_ok=True)
    joblib.dump(data, PREPARED_PATH)
    print(f"[main] Prepared data saved to {PREPARED_PATH}")


def run_select(args):
    """Step 2: select the variables common to ANOVA and Lasso."""
    data = _load_artifact(PREPARED_PATH, "prepare")
    features_sel = select_features(data["X_train"], data["y_train"], data["groups_train"])
    joblib.dump(features_sel, FEATURES_PATH)
    print(f"[main] {len(features_sel)} selected variables saved to {FEATURES_PATH}:")
    for f in features_sel:
        print(f"    - {f}")


def run_train(args):
    """Step 3: train the model (kept in artifacts/ until the 'save' step)."""
    data = _load_artifact(PREPARED_PATH, "prepare")
    features_sel = _load_artifact(FEATURES_PATH, "select")
    pipeline, columns = train_model(
        data["X_train"],
        data["y_train"],
        features_sel,
        groups_train=data["groups_train"],
        optimize=args.optimize,
    )
    joblib.dump((pipeline, columns), TRAINED_PATH)
    print(f"[main] Trained model kept in {TRAINED_PATH} (run 'python main.py save' next)")


def run_save(args):
    """Step 4: save the trained model as the final .pkl file."""
    pipeline, columns = _load_artifact(TRAINED_PATH, "train")
    save_model(pipeline, columns, args.model)


def run_load(args):
    """Step 5: load the saved model and show what it contains."""
    pipeline, columns = load_model(args.model)
    print(f"[main] Pipeline steps: {[name for name, _ in pipeline.steps]}")
    print(f"[main] The model needs {len(columns)} raw columns: {columns}")


def run_evaluate(args):
    """Step 6: load the saved model and evaluate it on the test set."""
    data = _load_artifact(PREPARED_PATH, "prepare")
    pipeline, columns = load_model(args.model)
    evaluate_model(pipeline, columns, data["X_test"], data["y_test"])


def run_predict(args):
    """Step 7: predict readmission for a few test patients."""
    data = _load_artifact(PREPARED_PATH, "prepare")
    pipeline, columns = load_model(args.model)
    X_new = data["X_test"].head(args.n_samples)
    result = predict(pipeline, columns, X_new)
    result["reel"] = data["y_test"].loc[result.index]
    result.to_csv(PREDICTIONS_PATH, index_label="index")
    print(result.to_string())
    print(f"[main] Predictions saved to {PREDICTIONS_PATH}")


STEPS = {
    "prepare": [run_prepare],
    "select": [run_select],
    "train": [run_train],
    "save": [run_save],
    "load": [run_load],
    "evaluate": [run_evaluate],
    "predict": [run_predict],
    "all": [run_prepare, run_select, run_train, run_save, run_evaluate, run_predict],
}


def main():
    parser = argparse.ArgumentParser(
        description="ML pipeline: hospital readmission of diabetic patients"
    )
    parser.add_argument("step", choices=list(STEPS), help="pipeline step to run")
    parser.add_argument(
        "--data", default="data/diabetic_data.csv", help="path to diabetic_data.csv"
    )
    parser.add_argument(
        "--model", default="models/modele_readmission_25.pkl", help="model file path"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="tune hyperparameters with RandomizedSearchCV (slow)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10, help="number of test patients to predict"
    )
    args = parser.parse_args()

    for step in STEPS[args.step]:
        step(args)


if __name__ == "__main__":
    main()
