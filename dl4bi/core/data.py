"""Data containers and multiprocessing dataloader helpers."""

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from queue import Empty
from typing import Callable

import jax
from jax import random


@dataclass(frozen=True)
class ElementSelectorMixin(Mapping):
    """Provide helpers for selecting a single element from batched mappings."""

    def element(self, i: int):
        """Extract the ``i``-th element from each array-valued field."""
        d = {}
        for k, v in self.items():
            d[k] = v
            if isinstance(v, jax.Array):
                d[k] = v[i]
        return d


@dataclass(frozen=True)
class Data(Mapping):
    """Immutable mapping-style container for dataset-level arrays."""

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def eq_shapes(self, other):
        """Return ``True`` when matching fields have compatible shapes."""
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
    """Immutable mapping-style container for batched model inputs."""

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def eq_shapes(self, other):
        """Return ``True`` when matching fields have compatible shapes."""
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


# TODO(danj): this copies CUDA state which reserved memory
# for dataloaders that don't require CUDA, but is required
# for multiprocessing with JAX...
# ctx = mp.get_context("spawn")


def dataloader_worker(
    rng: jax.Array,
    dataloader: Callable,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    """Run a dataloader in a worker process and stream batches to a queue."""
    batches = dataloader(rng)
    while True:
        try:
            cmd = input_queue.get(timeout=0.1)
            if cmd == "STOP":
                break
        except Empty:
            pass
        try:
            batch = next(batches)
            output_queue.put(batch)
        except Exception as e:
            output_queue.put(e)
            break


def multiprocess_dataloader(
    rng: jax.Array,
    dataloader: Callable,
    num_workers: int = 4,
    queue_size: int = 32,
):
    """Prefetch batches from a dataloader using multiple worker processes."""
    output_queue = ctx.Queue(maxsize=queue_size)
    input_queues = [ctx.Queue() for _ in range(num_workers)]
    workers = []
    for i in range(num_workers):
        rng_i = random.fold_in(rng, i)
        p = ctx.Process(
            target=dataloader_worker,
            args=(rng_i, dataloader, input_queues[i], output_queue),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    def generator():
        try:
            while True:
                result = output_queue.get()
                if isinstance(result, Exception):
                    raise result
                yield result
        finally:
            for q in input_queues:
                q.put("STOP")
            for w in workers:
                w.join()

    return generator()


def multiprocess_wrapper(
    dataloader: Callable,
    num_workers: int = 4,
    queue_size: int = 32,
):
    """Partially apply multiprocessing settings to a dataloader factory."""
    return partial(
        multiprocess_dataloader,
        dataloader=dataloader,
        num_workers=num_workers,
        queue_size=queue_size,
    )
