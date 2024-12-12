from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_mnist_dataset() -> DataLoader:
    """Downloads and loads the FashionMNIST dataset.

    Returns:
        DataLoader: A DataLoader for the FashionMNIST training data.
    """
    training_data: Dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=64)
    return train_dataloader
