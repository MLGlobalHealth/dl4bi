from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

import jax


@dataclass(frozen=True)
class ElementSelectorMixin(Mapping):
    def element(self, i: int):
        d = {}
        for k, v in self.items():
            d[k] = v
            if isinstance(v, jax.Array):
                d[k] = v[i]
        return d


@dataclass(frozen=True)
class Data(Mapping):
    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def eq_shapes(self, other):
        for f in fields(self):
            lhs = getattr(self, f.name)
            rhs = getattr(other, f.name)
            if isinstance(lhs, jax.Array):
                if lhs.shape != rhs.shape:
                    return False
            elif isinstance(lhs, (dict, set)):
                if set(lhs) != set(rhs):
                    return False
            elif isinstance(lhs, (list, tuple)):
                if len(lhs) != len(rhs):
                    return False
            if lhs is None:
                if rhs is not None:
                    return False
            return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        for k, v in self.items():
            if isinstance(v, jax.Array):
                if not (v == other[k]).all():
                    return False
            elif v != other[k]:
                return False
        return True

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


@dataclass(frozen=True)
class Batch(Mapping):
    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def eq_shapes(self, other):
        for f in fields(self):
            lhs = getattr(self, f.name)
            rhs = getattr(other, f.name)
            if isinstance(lhs, jax.Array):
                if lhs.shape != rhs.shape:
                    return False
            elif isinstance(lhs, (dict, set)):
                if set(lhs) != set(rhs):
                    return False
            elif isinstance(lhs, (list, tuple)):
                if len(lhs) != len(rhs):
                    return False
            elif lhs is None:
                if rhs is not None:
                    return False
            return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        for k, v in self.items():
            if isinstance(v, jax.Array):
                if not (v == other[k]).all():
                    return False
            elif v != other[k]:
                return False
        return True

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))
