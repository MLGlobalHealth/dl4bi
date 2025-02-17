from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

import jax


@dataclass(frozen=True)
class BatchElement:
    """Same as a batch but without the leading batch dim."""

    x_ctx: Optional[jax.Array] = None
    s_ctx: Optional[jax.Array] = None
    f_ctx: Optional[jax.Array] = None
    valid_lens_ctx: Optional[jax.Array] = None
    s_test: Optional[jax.Array] = None
    x_test: Optional[jax.Array] = None
    f_test: Optional[jax.Array] = None
    valid_lens_test: Optional[jax.Array] = None
    inv_permute_idx: Optional[jax.Array] = None

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None


@dataclass(frozen=True)
class Batch:
    """A batch."""

    x_ctx: Optional[jax.Array] = None
    s_ctx: Optional[jax.Array] = None
    f_ctx: Optional[jax.Array] = None
    valid_lens_ctx: Optional[jax.Array] = None
    x_test: Optional[jax.Array] = None
    s_test: Optional[jax.Array] = None
    f_test: Optional[jax.Array] = None
    valid_lens_test: Optional[jax.Array] = None
    inv_permute_idx: Optional[jax.Array] = None

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __getitem__(self, i: int):
        d = {}
        for k, v in self.items():
            if isinstance(v, jax.Array):
                if k.endswith(("ctx", "test")):
                    v = v[i]  # [B] or [B, L, D]
                elif k == "inv_permute_idx":
                    # inv_permute_idx shape meaning:
                    # [L]: A single permutation for all batch elements
                    # [B, L]: A separate permutation for each batch element
                    # [B, T, L]: A separate permutation for each batch element and timestep
                    if v.ndim > 1:
                        v = v[i]
            d[k] = v
        return BatchElement(**d)

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None
