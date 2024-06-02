import random

from torchvision.transforms import v2

from ....utils.torch.transforms import Shear, Translation, Rotation

_RANDOM_MIRROR = True


def shear_x(mag: float) -> Shear:  # [-0.3, 0.3]
    assert -0.3 <= mag <= 0.3
    if _RANDOM_MIRROR and random.random() > 0.5:
        mag = -mag
    return Shear([mag, 0.0])


def shear_y(mag: float) -> Shear:  # [-0.3, 0.3]
    assert -0.3 <= mag <= 0.3
    if _RANDOM_MIRROR and random.random() > 0.5:
        mag = -mag
    return Shear([0.0, mag])


def translate_x(mag: float) -> Translation:  # [-0.45, 0.45]
    assert -0.45 <= mag <= 0.45
    if _RANDOM_MIRROR and random.random() > 0.5:
        mag = -mag
    return Translation((mag, 0.0))


def translate_y(mag: float) -> Translation:  # [-0.45, 0.45]
    assert -0.45 <= mag <= 0.45
    if _RANDOM_MIRROR and random.random() > 0.5:
        mag = -mag
    return Translation((0.0, mag))


def rotate(mag: float) -> Rotation:  # [-30, 30]
    assert -30 <= mag <= 30
    if _RANDOM_MIRROR and random.random() > 0.5:
        mag = -mag
    return Rotation(mag, expand=True)


def auto_contrast(mag: float) -> v2.RandomAutocontrast:
    return v2.RandomAutocontrast(1.0)


def invert(mag: float) -> v2.RandomInvert:
    return v2.RandomInvert(1.0)


def equalize(mag: float) -> v2.RandomEqualize:
    return v2.RandomEqualize(1.0)


def flip(mag: float) -> v2.RandomHorizontalFlip:
    return v2.RandomHorizontalFlip(1.0)


def solarize(mag: float) -> v2.RandomSolarize:  # [0, 256]
    assert 0.0 <= mag <= 256.0
    return v2.RandomSolarize(mag, 1.0)


def posterize(mag: float) -> v2.RandomPosterize:  # [4, 8]
    bits = int(mag)
    assert 4 <= bits <= 8
    return v2.RandomPosterize(bits, 1.0)


def posterize_2(mag: float) -> v2.RandomPosterize:
    bits = int(mag)
    assert 4 <= bits <= 8  # [0, 4]
    return v2.RandomPosterize(bits, 1.0)


def contrast(mag: float) -> v2.ColorJitter:  # [0.1,1.9]
    assert 0.1 <= mag <= 1.9
    return v2.ColorJitter(contrast=(mag, mag))


def color(mag: float) -> v2.ColorJitter:  # [0.1,1.9]
    assert 0.1 <= mag <= 1.9
    return v2.ColorJitter(saturation=(mag, mag))


def brightness(mag: float) -> v2.ColorJitter:  # [0.1,1.9]
    assert 0.1 <= mag <= 1.9
    return v2.ColorJitter(brightness=(mag, mag))


def sharpness(mag: float) -> v2.RandomAdjustSharpness:
    assert 0.1 <= mag <= 1.9  # [0.1,1.9]
    return v2.RandomAdjustSharpness(mag, 1.0)


def cutout(mag: float) -> v2.RandomCrop:  # [0, 0.05]
    assert 0.0 <= mag <= 0.05
    return v2.RandomErasing(1.0, scale=(mag, mag), ratio=(1.0, 1.0), value=122)
