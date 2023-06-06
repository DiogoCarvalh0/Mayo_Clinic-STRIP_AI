from typing import Dict, List, Tuple

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float, float]: Train Loss, Accuracy and AUC.
    """
    model.train()

    train_loss, train_acc, train_auc = 0, 0, 0
    targets, predictions = [], []  # To calculate AUC

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X).flatten().type(torch.float64)
        targets.append(y)
        predictions.append(y_pred)

        loss = loss_fn(y_pred, y.type(torch.float64))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        y_pred_class = torch.round(torch.sigmoid(y_pred))
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # To calculate AUC
    targets = torch.cat(targets)
    predictions = torch.cat(predictions)
    train_auc = roc_auc_score(targets.detach().numpy(), predictions.detach().numpy())

    return train_loss, train_acc, train_auc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float, float]: Test Loss, Accuracy and AUC.
    """
    model.eval()

    test_loss, test_acc, test_auc = 0, 0, 0
    targets, predictions = [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X).flatten().type(torch.float64)
            targets.append(y)
            predictions.append(test_pred_logits)
            loss = loss_fn(test_pred_logits, y.type(torch.float64))
            test_loss += loss.item()

            test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)
    test_auc = roc_auc_score(targets.numpy(), predictions.numpy())

    return test_loss, test_acc, test_auc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils,
) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    It uses Reduce LR On Plateau regularization based on test loss.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model (torch.nn.Module): A PyTorch model to be trained and tested.
      train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
      test_dataloader ()torch.utils.data.DataLoader: A DataLoader instance for the model to be tested on.
      optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
      loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on both datasets.
      epochs (int): An integer indicating how many epochs to train for.
      device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").
      writer (torch.utils.tensorboard.writer.SummaryWriter): A SummaryWriter() instance to log model results to.
    """
    results = {"train_loss": [], "train_acc": [], "train_auc": [], "test_loss": [], "test_acc": [], "test_auc": []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_auc = train_step(
            model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device
        )

        test_loss, test_acc, test_auc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_auc: {train_auc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_auc: {test_auc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_auc"].append(train_auc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_auc"].append(test_auc)

        if writer:
            writer.add_scalar(tag="Loss/train", scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag="Loss/test", scalar_value=test_loss, global_step=epoch)
            writer.add_scalar(tag="Accuracy/train", scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag="Accuracy/test", scalar_value=test_acc, global_step=epoch)
            writer.add_scalar(tag="AUC/train", scalar_value=train_auc, global_step=epoch)
            writer.add_scalar(tag="AUC/test", scalar_value=test_auc, global_step=epoch)

            writer.close()

    return results
