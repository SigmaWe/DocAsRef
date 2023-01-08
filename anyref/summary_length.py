from datasets.arrow_dataset import Dataset
import numpy as np
import typing


def min(dataset: Dataset, ratio: float) -> int:
    return int(np.min(dataset["len"]) * ratio)


def mean(dataset: Dataset, ratio: float) -> int:
    return int(np.mean(dataset["len"]) * ratio)


def constant(_: typing.Optional[Dataset] = None, len: typing.Optional[int] = None) -> typing.Optional[int]:
    return len


def default(_: typing.Optional[Dataset] = None) -> None:
    return None
