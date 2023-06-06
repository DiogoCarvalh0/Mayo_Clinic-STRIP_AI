from typing import List, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import MayoClinicDataset


def create_weigthed_random_sampler(
    dataset: Union[MayoClinicDataset, pd.DataFrame], sample_weights: List[float] = None
) -> WeightedRandomSampler:
    """Creates WeightedRandomSampler to give same importance to all the classes in the dataset.

    Args:
        dataset (Union[MayoClinicDataset, pd.DataFrame]): MayoClinicDataset dataset or pandas.DataFrame to create one.
        sample_weights (List[float], optional): Weights to give to every example.
        If None then weights are calculated in order to give the same importance to all the classes in the dataset.
        Defaults to None.

    Returns:
        WeightedRandomSampler: Sampler
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = MayoClinicDataset(csv_file=dataset, root_dir="./data/images/", transform=None)

    if not sample_weights:
        class_appearances = dataset.tabular_data["label"].value_counts()
        class_weights = list(1 / class_appearances)

        sample_weights = [class_weights[label] for (data, label) in dataset]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloader(
    tabular_data: List[pd.DataFrame],
    img_root_dir: str,
    transforms: List[transforms.Compose],
    batch_size: int = 1,
    train_shuffle: bool = True,
    train_sampler: torch.utils.data.Sampler = None,
    num_workers: int = 0,
) -> Tuple[Tuple[torch.utils.data.DataLoader], List[str]]:
    """
    Creates Dataloaders for each tabular dataframe given, based on MayoClinicDataset. 
    It assumes the first dataframe given in the list is the train dataset.

    Args:
        tabular_data (List[pd.DataFrame]): List with the tabular data.
        img_root_dir (str): Path to the image folder.
        transforms (List[transforms.Compose]): List of transforms.Compose to indicate which tranformation to apply\
                                                to each dataframe.
        batch_size (int, optional): Number of samples per batch in each of the DataLoaders. Defaults to 1.
        train_shuffle (bool, optional): Whether or not to shuffle train data. Defaults to False.
        train_sampler (torch.data.utils.Sampler, optional): Sampler to be used for train Dataloader. Defaults to None.
        num_workers (int, optional): Number of workers (cpu's). Defaults to 0.

    Returns:
        Tuple[Tuple[torch.utils.data.DataLoader], List[str]]: Returns a Tuple with the same amount of DataLoaders\
            as the Tabular Data that was given, as well as and List with the target classes.
    """
    assert len(tabular_data) == len(
        transforms
    ), "Number of tabular data, must be the same as the number of transformation objects."

    train_shuffle = False if train_sampler else train_shuffle

    datasets = [MayoClinicDataset(data, img_root_dir, transform) for data, transform in zip(tabular_data, transforms)]

    dataloaders = []

    for i, dataset in enumerate(datasets):
        if i == 0:  # Train Dataser => create train DataLoader
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=train_shuffle,
                    sampler=train_sampler,
                    num_workers=num_workers,
                )
            )
        else:
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers=num_workers,
                )
            )

    return dataloaders, datasets[0].classes


def create_TTA_dataloader(
    tabular_data: Union[str, pd.DataFrame],
    img_root_dir: str,
    base_transforms: transforms.Compose,
    aug_transforms: transforms.Compose,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
    """
    Creates Dataloaders for the tabular dataframe given to be used with TTA predictions, based on MayoClinicDataset.

    Args:
        tabular_data (Union[str, pd.DataFrame]): Path or pandas Dataframe to tabular data.
        img_root_dir (str): Path to the image folder.
        base_transforms (transforms.Compose): Compose to indicate which tranformation, without any data augmentation.
        Ex: Resize, ToTensor,...
        aug_transforms (transforms.Compose): Compose to indicate which tranformation, with data augmentation steps.
        batch_size (int, optional): Number of samples per batch in each of the DataLoaders. Defaults to 1.
        num_workers (int, optional): Number of workers (cpu's). Defaults to 0.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]: Returns a Tuple with the
        base Dataloader, the augment Dataloader and the target classes labels
    """

    base_dataset = MayoClinicDataset(tabular_data, img_root_dir, base_transforms)

    classes = base_dataset.classes

    base_dataloader = DataLoader(
        dataset=base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    augment_dataloader = DataLoader(
        dataset=MayoClinicDataset(tabular_data, img_root_dir, aug_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return base_dataloader, augment_dataloader, classes
