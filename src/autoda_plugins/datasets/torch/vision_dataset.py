import os
from PIL import Image
from typing import Tuple, List, Optional, Callable, Any, Dict, Union

import torch
import torchvision.datasets.vision as vision

from torchvision import tv_tensors
from torchvision.transforms import v2

FilePathsType = Tuple[List[str], Union[str, List[str]]]


class AutoDaVisionDataset(vision.VisionDataset):  # type: ignore
    """
    AutoDaVisionDataset is an abstract class derived from Torchvision VisionDataset.

    It implements *__getitem__* and *__len__* with the default functionality to access
    images and targets of the dataset and the number of data points. This base class is
    not as flexible as Torchvision VisionDataset. However, if this base class is used,
    it is sufficient to specify the file paths to images and targets and how to load them.
    AutoDaVisionDataset takes care of the data access and the transform functions.

    If you derive from AutoDaVisionDataset, *load_image*, *load_target* and *file_paths*
    must be implemented. Also make sure that super() is called to initialize the
    described functionalities.
    """

    def __init__(
            self,
            root: str,
            base_dir: str,
            transforms: Optional[Callable[..., Any]] = None,
            transform: Optional[Callable[..., Any]] = None,
            target_transform: Optional[Callable[..., Any]] = None,
            use_transforms_v2: bool = True,
            bounding_box_format: tv_tensors.BoundingBoxFormat = tv_tensors.BoundingBoxFormat.XYXY
    ) -> None:
        """
        Initializes the AutoDaVisionDataset base class.

        - `root: str` Root directory of dataset where base_dir exists.

        - `base_dir: str` Name of the directory in which the dataset is saved.

        - `transforms: Callable = None` A function/transforms that takes in an image and a target
            and returns the transformed versions of both.

        - `transform: Callable = None` A function/transform that takes in a PIL image
            and returns a transformed version.

        - `target_transform: Callable = None` A function/transform that takes in the target and
            transforms it.

        - `use_transforms_v2: bool = True` If true, the image and the target are converted so that
            Torchvisions transforms v2 can be applied to them.

        - `convert_to_transforms_v2: bool = True` If true, the target is converted to be compatible
            with Torchvision Transforms v2. To find out more about this, see:
            https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#getting-started-with-transforms-v2

        - `bounding_box_format: tv_tensors.BoundingBoxFormat = tv_tensors.BoundingBoxFormat.XYXY`
            Determines the format of the bounding box. It is only relevant if
            convert_to_transforms_v2 is True.
        """
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.use_transforms_v2 = use_transforms_v2
        self.__base_dir = base_dir
        self.__img_files, self.__target_files = self.file_paths(self.dataset_path)
        self.__bounding_box_format = bounding_box_format

    @property
    def root(self) -> str:
        """
        Root directory of dataset where base_dir exists.
        """
        return self._root

    @root.setter
    def root(self, value: str) -> None:
        """
        Sets the root directory of the dataset.

        - `value: str` Path of the root directory.
        """
        self._root = os.path.abspath(value)

    @property
    def use_transforms_v2(self) -> bool:
        """
        If true, the image and the target are converted so that
        Torchvisions transforms v2 can be applied to them.
        """
        return self.__use_transforms_v2

    @use_transforms_v2.setter
    def use_transforms_v2(self, value: bool) -> None:
        """
        Sets whether the image and the target are converted for Torchvisions transforms v2.

        - `value: bool` Whether the image and the target are converted or not.
        """
        self.__use_transforms_v2 = value

    @property
    def base_dir(self) -> str:
        """
        Name of the directory in which the dataset is saved.
        """
        return self.__base_dir

    @property
    def dataset_path(self) -> str:
        """
        Returns the path to the directory that contains the dataset files.
        """
        return os.path.join(self.root, self.__base_dir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns the image and the target at the specified index.

        - `index: int` Index of the image and target.
        """
        img = self.load_image(index)
        target = self.load_target(index)

        if self.use_transforms_v2:
            img, target = self.__transforms_v2_converte(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        """
        Returns the number of datapoints.
        """
        return len(self.__img_files)

    def __transforms_v2_converte(
        self,
        img: Image.Image,
        target: Dict[str, Any]
    ) -> Tuple[tv_tensors.Image, Dict[str, Any]]:
        """
        Converts the image and the target so that Torchvisions transforms v2 can be applied to them.

        - `img: Image.Image`: The image to be converted.

        - `target: Dict[str, Any]`: The target to be converted.

        Returns the converted image and target.
        """
        img_v2 = tv_tensors.Image(img)
        img_v2 = v2.functional.to_dtype(img_v2, dtype=torch.float32, scale=True)

        if 'boxes' in target:
            target['boxes'] = tv_tensors.BoundingBoxes(
                target['boxes'],
                format=self.__bounding_box_format,
                canvas_size=img_v2.shape[-2:]
            )

        return img_v2, target

    def __target_file(self, index: int) -> str:
        """
        Checks whether there is a single target file or a list of target files.
        If there is a single file, it is returned directly.
        Otherwise, the target file at the specified index is returned.
        """
        if isinstance(self.__target_files, str):
            return self.__target_files
        return self.__target_files[index]

    def load_image(self, index: int) -> Any:
        """
        Implement this method so that it loads and returns the specified image file.

        - `index: int`: Index of the image file to load.

        This method must be overloaded in your concrete class, otherwise NotImplementedError
        is raised.

        Returns the image loaded into the memory.
        """
        raise NotImplementedError

    def load_target(self, index: int) -> Dict[str, Any]:
        """
        Implement this method so that it loads the specified target file. The contents of
        the file should be transferred to a dict in order to return the target information.

        - `index: int`: Index of the target file to load.

        This method must be overloaded in your concrete class, otherwise NotImplementedError
        is raised.

        Returns the target information.
        """
        raise NotImplementedError

    def file_paths(self, dataset_path: str) -> FilePathsType:
        """
        Implement this method so that it returns the file paths of the images and targets in
        two separate lists. Make sure that the indices of the image and the target file
        path correspond. If there is only a single target file path, return it as str.
        The file paths must be relative to the specified dataset_path.

        - `dataset_path: str` The path to the directory that contains the dataset files.

        This method must be overloaded in your concrete class, otherwise NotImplementedError
        is raised.

        Returns a tuple. The first entry is a list containing the paths of the images and the
        second entry is a single target file path or a list of all target file paths.
        """
        raise NotImplementedError

    def image_path(self, index: int) -> str:
        """
        Returns the image file path at the specified index.
        """
        return os.path.join(self.dataset_path, self.__img_files[index])

    def target_path(self, index: int) -> str:
        """
        Returns the target file path at the specified index.
        """
        return os.path.join(self.dataset_path, self.__target_file(index))
