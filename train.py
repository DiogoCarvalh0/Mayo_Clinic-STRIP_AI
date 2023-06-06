import time
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch import nn, optim

import src.model_builder as model_builder
import src.utils as utils
from src.data_setup import (
    create_dataloader,
    create_TTA_dataloader,
    create_weigthed_random_sampler,
)
from src.predict import predict, predict_TTA
from src.train_procedures import train


def plot_ROC_Curve(fpr: List, tpr: List, auc: float, cm, ax) -> None:
    """Plots ROC Curve.

    Args:
        fpr (List): False Positive Rate points.
        tpr (List): True Positive Rate points.
        auc (float): Model AUC.
        cm (_type_): Model Confusion Matrix.
        ax (_type_): Axis to plot.
    """
    tpr_score = float(cm[1][1]) / (cm[1][1] + cm[1][0])
    fpr_score = float(cm[0][1]) / (cm[0][0] + cm[0][1])

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.plot(fpr_score, tpr_score, marker="o", color="black")

    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")

    ax.legend(loc="lower right")

    return ax


VAL_SIZE = 0.1
# TEST_SIZE = 0.1
TEST_CENTER_ID = 7
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 3e-3

# MODEL = "efficientnet_b0-Dropout-Linear1"
MODEL = "convnext_base"
MODEL_NAME = f"{MODEL}-{time.time()}"
MODEL_SAVE_EXTENSION = ".pt"  # .pt or .pth

EXPERIMENT_NAME = f"{MODEL}"
MODEL_SAVE_DIR = "./models/"

CALC_MEAN_STD = False

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # MPS give an error
DEVICE = "cpu"

if __name__ == "__main__":
    train_tabular_data, val_tabular_data, test_tabular_data = utils.train_val_test_split_with_center_id(
        csv_file="./data/train.csv", val_split=VAL_SIZE, test_center_id=TEST_CENTER_ID
    )

    train_data_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.9454, 0.8770, 0.8563), std=(0.1034, 0.2154, 0.2716)),
        ]
    )

    eval_data_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.9454, 0.8770, 0.8563), std=(0.1034, 0.2154, 0.2716)),
        ]
    )

    sampler = create_weigthed_random_sampler(dataset=train_tabular_data)

    (train_dataloader, val_dataloader, test_dataloader), classes = create_dataloader(
        tabular_data=[train_tabular_data, val_tabular_data, test_tabular_data],
        img_root_dir="./data/images/",
        transforms=[train_data_transform, eval_data_transform, eval_data_transform],
        batch_size=BATCH_SIZE,
        train_sampler=sampler,
    )

    model = model_builder.create_convnext_model("b")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    writer = utils.create_writer(experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
        writer=writer,
    )

    utils.save_model(model=model, target_dir=MODEL_SAVE_DIR, model_name=f"{MODEL_NAME}{MODEL_SAVE_EXTENSION}")

    # Test predictions
    test_predictions, y_test = predict(model=model, dataloader=test_dataloader, device=DEVICE)

    auc = roc_auc_score(y_test.numpy(), test_predictions.numpy())
    class_test_predictions = pd.DataFrame({"CE": 1 - test_predictions.numpy(), "LAA": test_predictions.numpy()})
    bce_loss = loss_fn(test_predictions, y_test.type(torch.float64))
    multi_class_log_loss = utils.multi_class_logarithmic_loss(y_test.numpy(), class_test_predictions.to_numpy())

    print(
        f"Accuracy = {torch.sum(y_test == torch.round(test_predictions).type(torch.int))/len(test_predictions)*100:.2f}"
    )
    print(f"AUC = {auc:.3f}")
    print(f"Binary Cross Entropy = {bce_loss:.5f}")
    print(f"Multi-class logarithmic loss = {multi_class_log_loss:.5f}")

    # Test Predictions with TTA
    base_test_dataloader, aug_test_dataloader, classes = create_TTA_dataloader(
        tabular_data=test_tabular_data,
        img_root_dir="./data/images/",
        base_transforms=eval_data_transform,
        aug_transforms=train_data_transform,
        batch_size=BATCH_SIZE,
    )

    test_predictions_TTA, y_test = predict_TTA(
        model=model, base_dataloader=base_test_dataloader, aug_dataloader=aug_test_dataloader, device=DEVICE
    )

    auc_TTA = roc_auc_score(y_test.numpy(), test_predictions_TTA.numpy())
    class_test_predictions_TTA = pd.DataFrame(
        {"CE": 1 - test_predictions_TTA.numpy(), "LAA": test_predictions_TTA.numpy()}
    )
    bce_loss_TTA = loss_fn(test_predictions_TTA, y_test.type(torch.float64))
    multi_class_log_loss_TTA = utils.multi_class_logarithmic_loss(
        y_test.numpy(), class_test_predictions_TTA.to_numpy()
    )

    print(
        f"Accuracy with TTA = {torch.sum(y_test == torch.round(test_predictions_TTA).type(torch.int))/len(test_predictions_TTA)*100:.2f}"
    )
    print(f"AUC with TTA = {auc_TTA:.3f}")
    print(f"Binary Cross Entropy = {bce_loss_TTA:.5f}")
    print(f"Multi-class logarithmic loss with TTA = {multi_class_log_loss_TTA:.5f}")

    writer.add_scalar(tag="AUC/RealTest", scalar_value=auc_TTA, global_step=0)
    writer.add_scalar(tag="Loss/RealTest", scalar_value=bce_loss_TTA, global_step=0)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    cm_TTA = confusion_matrix(y_test.numpy(), torch.round(test_predictions_TTA).type(torch.int).numpy())
    fpr_TTA, tpr_TTA, thresholds_TTA = roc_curve(y_test.numpy(), test_predictions_TTA.numpy())

    fig, ax = plt.subplots(1, 2, figsize=(19, 9))

    sns.heatmap(
        cm_TTA,
        annot=True,
        cmap="viridis",
        linewidths=0.5,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax[0],
    )

    ax[0].set_ylabel("True label")
    ax[0].set_xlabel("Predicted label")

    plot_ROC_Curve(fpr_TTA, tpr_TTA, auc_TTA, cm_TTA, ax[1])

    plt.show()
