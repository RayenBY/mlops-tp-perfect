"""
pipeline_prefect.py
===================
Prefect pipeline (Atelier 3): each task calls a step of main.py or a code tool,
and flows group the tasks.

Usage (inside the venv, from the project folder):
    python pipeline_prefect.py --flow all          # whole pipeline
    python pipeline_prefect.py --flow install      # install the dependencies
    python pipeline_prefect.py --flow code         # format + quality + security + tests
    python pipeline_prefect.py --flow train        # prepare data + train + save the model
    python pipeline_prefect.py --flow entrainement # same as 'train'
    python pipeline_prefect.py --flow evaluate     # load the model + evaluate it
    python pipeline_prefect.py --flow predict      # load the model + predict
    python pipeline_prefect.py --flow clean        # remove generated files
    python pipeline_prefect.py --flow train --optimize   # train with hyperparameter tuning
"""

import argparse
import os
import shutil

# subprocess only runs fixed commands built in this file (no shell, no user input)
import subprocess  # nosec B404
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # the interpreter running this file (the venv's python)
MAIN_PY = "main.py"
REQUIREMENTS = "requirements.txt"

# Quality and security tools only look at our own files, not the venv or the data
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TESTS_DIR = "tests"


def _run(cmd):
    """Run a command in the project folder, send its output to the Prefect logs,
    and fail the task if the command fails."""
    logger = get_run_logger()
    logger.info("$ %s", " ".join(cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    # cmd is a list built in this file and run without a shell
    with subprocess.Popen(  # nosec B603
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        for line in proc.stdout:
            logger.info(line.rstrip())
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit code {returncode}): {' '.join(cmd)}")


def _main(step, *extra):
    """Run one step of main.py."""
    _run([PYTHON, MAIN_PY, step, *extra])


# ----------------------------------------------------------------------------
# Tasks: project and environment
# ----------------------------------------------------------------------------
@task(name="fetch-project")
def fetch_project():
    """Get the latest version of the project from the remote git repository."""
    logger = get_run_logger()
    if not (ROOT / ".git").exists():
        logger.warning("Not a git repository: skipping 'git pull'.")
        return
    if "[remote " not in (ROOT / ".git" / "config").read_text(encoding="utf-8"):
        logger.warning("No git remote configured: skipping 'git pull'.")
        return
    _run(["git", "pull", "--ff-only"])


@task(name="install-dependencies", retries=2, retry_delay_seconds=10)
def install_dependencies():
    """Install the packages listed in requirements.txt."""
    _run([PYTHON, "-m", "pip", "install", "-q", "-r", REQUIREMENTS])
    get_run_logger().info("Dependencies installed.")


# ----------------------------------------------------------------------------
# Tasks: code checks
# ----------------------------------------------------------------------------
@task(name="format-code")
def format_code():
    """Format the code with black."""
    _run([PYTHON, "-m", "black", *CODE_FILES, TESTS_DIR])


@task(name="check-quality")
def check_quality():
    """Check the code style and common errors with flake8."""
    _run([PYTHON, "-m", "flake8", *CODE_FILES, TESTS_DIR])
    get_run_logger().info("flake8: no problem found.")


@task(name="check-security")
def check_security():
    """Look for security issues with bandit."""
    _run([PYTHON, "-m", "bandit", "-q", *CODE_FILES])
    get_run_logger().info("bandit: no security issue found.")


@task(name="run-tests")
def run_tests():
    """Run the unit tests with pytest."""
    _run([PYTHON, "-m", "pytest", TESTS_DIR])


# ----------------------------------------------------------------------------
# Tasks: data and model (each one calls a step of main.py)
# ----------------------------------------------------------------------------
@task(name="prepare-data")
def prepare_data():
    """Load, clean and split the data."""
    _main("prepare")
    get_run_logger().info("Data preparation complete.")


@task(name="select-features")
def select_features():
    """Select the variables kept by both ANOVA and Lasso."""
    _main("select")


@task(name="train-model")
def train_model(optimize: bool = False):
    """Train the XGBoost model."""
    _main("train", *(["--optimize"] if optimize else []))


@task(name="save-model")
def save_model():
    """Save the trained model to models/."""
    _main("save")


@task(name="load-model")
def load_model():
    """Check that the saved model can be loaded."""
    _main("load")


@task(name="evaluate-model")
def evaluate_model():
    """Evaluate the saved model on the test set."""
    _main("evaluate")


@task(name="predict")
def predict(n_samples: int = 10):
    """Predict readmission for a few test patients."""
    _main("predict", "--n-samples", str(n_samples))


# ----------------------------------------------------------------------------
# Flows
# ----------------------------------------------------------------------------
@flow(name="install", log_prints=True)
def flow_install():
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code():
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="prepare", log_prints=True)
def flow_prepare():
    prepare_data()
    select_features()


@flow(name="train", log_prints=True)
def flow_train(optimize: bool = False):
    prepare_data()
    select_features()
    train_model(optimize)
    save_model()


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    load_model()
    evaluate_model()


@flow(name="predict", log_prints=True)
def flow_predict(n_samples: int = 10):
    predict(n_samples)


@flow(name="all", log_prints=True)
def flow_all(optimize: bool = False):
    fetch_project()
    install_dependencies()
    # code checks, right after the dependencies and before the data
    format_code()
    check_quality()
    check_security()
    run_tests()
    # data and model
    prepare_data()
    select_features()
    train_model(optimize)
    save_model()
    load_model()
    evaluate_model()
    predict()


@flow(name="clean", log_prints=True)
def flow_clean():
    """Remove generated files (the saved model in models/ is kept)."""
    for folder in ["artifacts", ".pytest_cache", "__pycache__", f"{TESTS_DIR}/__pycache__"]:
        path = ROOT / folder
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path}")


FLOWS = {
    "install": flow_install,
    "code": flow_code,
    "prepare": flow_prepare,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "all": flow_all,
    "clean": flow_clean,
}


def main():
    parser = argparse.ArgumentParser(description="Prefect ML pipeline")
    parser.add_argument("--flow", choices=list(FLOWS), default="all", help="flow to run")
    parser.add_argument(
        "--optimize", action="store_true", help="tune hyperparameters (train and all flows)"
    )
    args = parser.parse_args()

    if args.flow in ("train", "entrainement", "all"):
        FLOWS[args.flow](optimize=args.optimize)
    else:
        FLOWS[args.flow]()


if __name__ == "__main__":
    main()
