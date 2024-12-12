import mlflow
from mlflow.tracking import MlflowClient

from config import model, params
from pprint import pprint
import pandas as pd


from src1.prepare_data import get_iris_dataset

def yield_artifacts(run_id, path=None):
    """
    Recursively yield all artifact paths for a specified MLflow run.

    Args:
        run_id (str): The unique identifier of the MLflow run.
        path (str, optional): A specific path within the artifact repository.
            If not provided, the root directory is used.

    Yields:
        str: Paths to the artifacts stored in the run's artifact repository.

    This function uses the MLflow client to traverse the artifact directory structure
    and yields paths to all individual artifacts. If an artifact is a directory,
    the function recursively explores its contents.
    """
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """
    Fetch logged parameters, metrics, tags, and artifact paths from an MLflow run.

    Args:
        run_id (str): The unique identifier of the MLflow run.

    Returns:
        dict: A dictionary containing the following keys:
            - "params": A dictionary of parameters logged to the run.
            - "metrics": A dictionary of metrics logged to the run.
            - "tags": A dictionary of user-defined tags (excluding system tags).
            - "artifacts": A list of paths to artifacts stored in the run's artifact repository.

    This function uses the MLflow client to retrieve logged data from the specified run.
    System tags (e.g., tags starting with "mlflow.") are excluded from the returned tags.
    """
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }

def start_train_with_mlflow():
    mlflow.sklearn.autolog()
    train_data = get_iris_dataset()

    model.fit(train_data.data, train_data.target)
    # Retrieve the MLflow run ID of the parent run (created by `autolog`)
    run_id = mlflow.last_active_run().info.run_id
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print(f"\n---------- logged {key} ----------")
        pprint(data)
    filter_child_runs = f"tags.mlflow.parentRunId = '{run_id}'"
    runs = mlflow.search_runs(filter_string=filter_child_runs)

    # Extract specific columns for display:
    # - `params.kernel` and `params.C`: The hyperparameters for each child run.
    # - `metrics.mean_test_score`: The average test score for each parameter combination.
    param_cols = [f"params.{p}" for p in params.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # Prevent truncating columns in the output
    print(runs[["run_id", *param_cols, *metric_cols]])

start_train_with_mlflow()

print("End")
