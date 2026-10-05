from abc import ABC, abstractmethod
from typing import Any, Literal

from torch.utils.data import Dataset

Modality = Literal["image", "text", "audio"]
TaskType = Literal["transmission", "classification"]
Split = Literal["train", "val", "test"]


class SemComDataset(Dataset, ABC):
    """Common interface for all modalities: __getitem__ returns
    {"input": ..., "label": Optional[...], "raw": Optional[...]}.
    "input" dtype varies by modality (Tensor for image/audio, str for text) —
    tokenization/feature extraction happens in the Week 4 encoders, not here.
    """

    modality: Modality
    task_type: TaskType

    def __init__(self, split: Split):
        self.split = split

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...
