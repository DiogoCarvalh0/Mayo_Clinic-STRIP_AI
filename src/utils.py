from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """
    import os
    from datetime import datetime

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

    return SummaryWriter(log_dir=log_dir)


def get_mean_and_std(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and std over every channel in the image.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple with the mean, std of every image channel.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def train_val_test_split(
    csv_file: str, val_split: float, test_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divides tabular data (csv file) into train, validation and test.

    Args:
        csv_file (str): Path to csv file.
        val_split (float): Percentage of data to be used for validation.
        test_split (float): Percentage of data to be used for test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test dataframes.
    """
    assert (test_split > 0 and test_split < 1) and (
        val_split > 0 and val_split < 1
    ), "val_size and test_size have to be between 0 and 1."

    assert val_split + test_split < 1, "Validation and Test need to sum up to something less than 1."

    data = pd.read_csv(csv_file)
    nr_exaples = len(data)
    nr_val_examples = int(nr_exaples * val_split)
    nr_test_examples = int(nr_exaples * test_split)

    train_indices, test_indices = train_test_split(
        list(range(nr_exaples)),
        test_size=nr_test_examples,
        stratify=data["label"].to_numpy(),
    )

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=nr_val_examples,
        stratify=data["label"].iloc[train_indices].to_numpy(),
    )

    return data.iloc[train_indices], data.iloc[val_indices], data.iloc[test_indices]


def train_val_test_split_with_center_id(
    csv_file: str, val_split: float, test_center_id: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divides tabular data (csv file) into train, validation and test.
    Test data is all the center_id examples.

    Args:
        csv_file (str): Path to csv file.
        val_split (float): Percentage of data to be used for validation.
        test_center_id (int): Center ID to save for test data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test dataframes.
    """
    assert val_split > 0 and val_split < 1, "val_size and test_size have to be between 0 and 1."

    data = pd.read_csv(csv_file)

    test_data = data.loc[data["center_id"] == test_center_id]
    non_test_data = data.loc[data["center_id"] != test_center_id]

    train_data, val_data = train_test_split(
        non_test_data,
        test_size=val_split,
        stratify=non_test_data["label"].to_numpy(),
    )

    return train_data, val_data, test_data


def multi_class_logarithmic_loss(y_true: List[int], y_preds: Tuple[List]) -> float:
    """
    Calculates the weighted multi-class logarithmic loss.
    The code is not optimized as it assumes the weights for each class to be the same (1/nr_classes),
    however it does not take this assumpytion into consideration when calculating the loss.

    Args:
        y_true (List[int]): True values.
        y_preds (Tuple[List]): Predicted values

    Returns:
        float: Mean loss value.
    """
    loss = 0

    if not isinstance(y_true, (np.ndarray, np.generic)):
        y_true = np.array(y_true)

    if not isinstance(y_preds, (np.ndarray, np.generic)):
        y_preds = np.array(y_preds)

    classes, counts = np.unique(y_true, return_counts=True)
    nr_classes = len(classes)

    # Gives same weight to every class 1/number of class
    w = np.zeros(nr_classes) + 1 / nr_classes

    # Normalize predictions
    y_preds = y_preds / np.expand_dims(np.sum(y_preds, axis=1), axis=-1)

    # Clip predicted probabilities
    y_preds = np.clip(y_preds, 10**-15, 1 - 10**-15)

    for true, preds in zip(y_true, y_preds):
        for i in range(nr_classes):
            if true != classes[i]:
                continue  # When it is not the true class the value added is 0, so we ignore it

            loss += -((w[i] * np.log(preds[i]) / counts[i]) / (np.sum(w)))

    return loss / len(y_true)
