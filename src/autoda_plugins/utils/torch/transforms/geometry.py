from typing import List, Any, Sequence, Union, Optional, Dict, Type

from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional._utils import _FillType


class Shear(v2.RandomAffine):
    def __init__(
        self,
        shear: Sequence[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
        center: Optional[List[float]] = None
    ):
        super().__init__(
            degrees=0,
            shear=[shear[0], shear[0], shear[1], shear[1]],
            interpolation=interpolation,
            fill=fill,
            center=center
        )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(
            angle=0.0,
            translate=(0, 0),
            scale=1.0,
            shear=(self.shear[0], self.shear[2])
        )


class Translation(v2.RandomAffine):
    def __init__(
        self,
        translate: Sequence[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
        center: Optional[List[float]] = None
    ):
        self.mirror_x = True if translate[0] < 0.0 else False
        self.mirror_y = True if translate[1] < 0.0 else False

        pos_translate = (
            -translate[0] if self.mirror_x else translate[0],
            -translate[1] if self.mirror_y else translate[1]
        )

        super().__init__(
            degrees=0,
            translate=pos_translate,
            interpolation=interpolation,
            fill=fill,
            center=center
        )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        translate = (
            -self.translate[0] if self.mirror_x else self.translate[0],
            -self.translate[1] if self.mirror_y else self.translate[1]
        )

        return dict(
            angle=0.0,
            translate=translate,
            scale=1.0,
            shear=(0.0, 0.0)
        )


class Rotation(v2.RandomRotation):
    def __init__(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0
    ):
        self.mirror = True if angle < 0.0 else False

        super().__init__(
            degrees=-angle if self.mirror else angle,
            interpolation=interpolation,
            expand=expand,
            center=center,
            fill=fill
        )

        self.angle = angle

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(angle=-self.angle if self.mirror else self.angle)
