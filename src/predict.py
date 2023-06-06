from typing import Tuple

import torch
from tqdm import tqdm


def predict(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Makes predictions for all the samples in the DataLoader.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader with data to predict (data should not be shuffled).
        device (torch.device): Device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predictions of the probabilities and observerd classes.
    """
    model.to(device)
    model.eval()

    ys = []
    predictions = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X).flatten().type(torch.float64)

            ys.append(y)
            predictions.append(torch.sigmoid(test_pred_logits))

    return torch.cat(predictions), torch.cat(ys)


def predict_TTA(
    model: torch.nn.Module,
    base_dataloader: torch.utils.data.DataLoader,
    aug_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_aug_samples: int = 4,
    beta: float = 0.25,
    use_max: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses Test Time Augmentation to make predictions.

    Args:
        model (torch.nn.Module): Trained model.
        base_dataloader (torch.utils.data.DataLoader): DataLoader with data to predict, without the data augmentation
        steps (data should not be shuffled).
        aug_dataloader (torch.utils.data.DataLoader): DataLoader with data to predict, with the data augmentation
        steps (data should not be shuffled).
        device (torch.device): Device.
        n_aug_samples (int, optional): Number of times to transform the data and make predictions on it. Defaults to 4.
        beta (float, optional): Importance given to prediction of augmented predictions. Defaults to 0.25.
        use_max (bool, optional): Whether or not to use maximum values for predictions. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predictions of the probabilities and observed classes.
    """
    model.to(device)
    model.eval()

    aug_predictions = []

    with tqdm(total=n_aug_samples + 1) as pbar:
        predictions, targets = predict(model, base_dataloader, device)

        pbar.update()

        for _ in range(n_aug_samples):
            aug_predictions.append(predict(model, aug_dataloader, device)[0])
            pbar.update()

    aug_predictions = torch.stack(aug_predictions)
    aug_predictions = aug_predictions.max(0)[0] if use_max else aug_predictions.mean(0)

    if use_max:
        return targets, torch.stack([predictions, aug_predictions], 0).max(0)[0]
    predictions = torch.lerp(aug_predictions, predictions, beta)

    return predictions, targets
