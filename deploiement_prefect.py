"""
deploiement_prefect.py
======================
Registers the flows of pipeline_prefect.py as Prefect deployments and runs a
local worker that executes them (on schedule or when triggered).

Schedules (Africa/Tunis time):
    ml-pipeline-all   -> every day at 02:00 (whole pipeline)
    ml-pipeline-code  -> every day at 01:00 (format, quality, security, tests)
    the other deployments only run when triggered manually.

How to test it (3 terminals, venv activated, from the project folder):
    Terminal 1:  prefect server start                        # UI on http://localhost:4200
    Terminal 2:  prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
                 python deploiement_prefect.py               # worker, keep it running
    Terminal 3:  prefect deployment ls
                 prefect deployment run 'all/ml-pipeline-all'
                 prefect deployment run 'train/ml-pipeline-train'
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import (
    flow_all,
    flow_clean,
    flow_code,
    flow_evaluate,
    flow_install,
    flow_predict,
    flow_train,
)

TIMEZONE = "Africa/Tunis"


def daily(hour, minute=0):
    """Schedule that runs once a day at hour:minute (Tunis time)."""
    return [CronSchedule(cron=f"{minute} {hour} * * *", timezone=TIMEZONE)]


if __name__ == "__main__":
    serve(
        flow_all.to_deployment(
            name="ml-pipeline-all",
            schedules=daily(2),
            tags=["mlops", "full-pipeline"],
            description="Whole pipeline: git pull, install, code checks, data, model.",
        ),
        flow_code.to_deployment(
            name="ml-pipeline-code",
            schedules=daily(1),
            tags=["mlops", "code"],
            description="Format, quality, security and unit tests.",
        ),
        flow_install.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
        flow_train.to_deployment(name="ml-pipeline-train", tags=["mlops", "training"]),
        flow_evaluate.to_deployment(name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]),
        flow_predict.to_deployment(name="ml-pipeline-predict", tags=["mlops", "prediction"]),
        flow_clean.to_deployment(name="ml-pipeline-clean", tags=["mlops", "maintenance"]),
    )
