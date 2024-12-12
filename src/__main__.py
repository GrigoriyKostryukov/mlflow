import logging

from torchinfo import summary

import mlflow
from config import device, epochs, loss_fn, metric_fn, model, optimizer
from prepare_data import get_mnist_dataset
from train import train

print('Start')
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("/mlflow-pytorch-quickstart")


def setup_logging(
    log_file: str = "training.log", level: int = logging.INFO
) -> None:
    """Sets up logging to a file."""
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",  # Overwrite the log file each time
    )


def start_train_with_mlflow():
    setup_logging()
    train_dataloader = get_mnist_dataset()

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "SGD",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for t in range(epochs):
            print(f"Epoch " f"{t + 1}\n-------------------------------")
            logging.info(f"Epoch " f"{t + 1}\n-------------------------------")
            train(
                train_dataloader,
                model,
                loss_fn,
                metric_fn,
                optimizer,
                device,
            )

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")


start_train_with_mlflow()

print("End")
