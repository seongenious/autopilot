"""Base dataset class for autonomous driving datasets."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for autonomous driving datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        data_root: Root directory of the dataset.
        ann_file: Path to annotation file.
        pipeline: List of transforms to apply.
        test_mode: Whether in test mode (no augmentation).
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: Optional[List[Callable]] = None,
        test_mode: bool = False,
    ) -> None:
        """Initialize BaseDataset."""
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.pipeline = pipeline or []
        self.test_mode = test_mode

        # Load annotations
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load dataset annotations.

        Returns:
            List of annotation dictionaries.
        """

    @abstractmethod
    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Get data info for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing sample information.
        """

    def prepare_data(self, idx: int) -> Dict[str, Any]:
        """Prepare data for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Processed data dictionary.
        """
        data_info = self.get_data_info(idx)

        # Apply transforms
        for transform in self.pipeline:
            data_info = transform(data_info)

        return data_info

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        return self.prepare_data(idx)


# TODO: Add support for dataset concatenation
# TODO: Add support for dataset sampling strategies
# TODO: Implement caching mechanism for faster data loading
