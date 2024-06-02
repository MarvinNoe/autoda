from typing import Tuple, Callable, List

from torchvision.transforms import v2


class AutoDaTransform():
    def __init__(
        self,
        transform_id: int,
        prob: float,
        level: float
    ):
        self.transform_id = transform_id
        self.prob = prob
        self.level = level

    def torchvision_random_apply_v2(
        self,
        transforms: List[Tuple[Callable, float, float]]
    ) -> v2.RandomApply:
        op_func, min_mag, max_mag = transforms[self.transform_id]
        mag = self.mag(min_mag, max_mag)
        return v2.RandomApply([op_func(mag)], p=self.prob)

    def mag(self, min_mag, max_mag) -> float:
        return self.level * (max_mag - min_mag) + min_mag

    def __repr__(self):
        op_str = f'(transform_id={self.transform_id}, prob={self.prob}, level={self.level})'
        return f'{type(self).__name__}: {v2.RandomApply.__name__} ( {op_str} )'


class AutoDaSubPolicy():
    def __init__(
        self
    ):
        self.transforms = []

    def append_transform(self, transform: AutoDaTransform):
        self.transforms.append(transform)

    def torchvision_compose_v2(self, transforms: List[Tuple[Callable, float, float]]) -> v2.Compose:
        transforms_v2 = [transform.torchvision_random_apply_v2(transforms)
                         for transform in self.transforms]
        return v2.Compose(transforms_v2)

    def __repr__(self):
        repr_indent = 4
        classname = f'{type(self).__name__}: {v2.Compose.__name__}'

        lines = [classname] + [' ' * repr_indent +
                               repr(sub_policy) for sub_policy in self.transforms]
        return '\n'.join(lines)


class AutoDaPolicy():
    def __init__(
        self
    ):
        self.sub_policies = []

    def append_sub_policy(self, sub_policy: AutoDaSubPolicy):
        self.sub_policies.append(sub_policy)

    def torchvision_random_choice_v2(
        self,
        transforms: List[Tuple[Callable, float, float]]
    ) -> v2.RandomChoice:
        sub_policies_v2 = [sub_policy.torchvision_compose_v2(
            transforms) for sub_policy in self.sub_policies]
        return v2.RandomChoice(sub_policies_v2)

    def __repr__(self):
        repr_indent = 4
        classname = f'{type(self).__name__}: {v2.RandomChoice.__name__}'
        lines = [classname]
        for sub_pol in self.sub_policies:
            sub_pol_lines = [' ' * repr_indent + line for line in repr(sub_pol).split('\n')]
            lines.extend(sub_pol_lines)
        return '\n'.join(lines)
