from collections import defaultdict
from pathlib import Path
from typing import Generator

import jax
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dl4bi.meta_learning.autoregressive import AutoregressiveSampler, Strategy
from dl4bi.meta_learning.train_utils import build_gp_dataloader, load_ckpt


def evaluate(
    rng,
    model: AutoregressiveSampler,
    dataloader: Generator,
    strategies: list[Strategy],
    num_samples_for_random: int,
):
    nlls = defaultdict(list)

    for i, datum in (pbar := tqdm(enumerate(dataloader), desc="Evaluating")):
        s_ctx, f_ctx, s_test, f_test, valid_lens_ctx = datum
        for strategy in strategies:
            rng, rng_i = jax.random.split(rng)
            nll = -model.logpdf(
                rng_i,
                s_ctx,
                f_ctx,
                s_test,
                f_test,
                valid_lens_ctx,
                strategy,
                num_samples_for_random,
            )
            nlls[strategy].append(nll)

        pbar.set_postfix(
            {f"NLL {strategy}": np.mean(nlls[strategy]) for strategy in strategies}
        )


# TODO @pgrynfelder: add other dataloaders
def dataloader(rng, data, kernel, N):
    gp_dataloader = build_gp_dataloader(data, kernel)(rng)

    for i, datum in enumerate(gp_dataloader):
        if i >= N:
            raise StopIteration

        s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, var, ls, period = datum

        # note that s, f come directly from the GP not the observation process
        # TODO @pgrynfelder: might be desirable to change this
        yield s_ctx, f_ctx, s, f, valid_lens_ctx


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=1000, help="size of the test dataset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the batch size set in config",
    )

    args = parser.parse_args()

    state, config = load_ckpt(args.model)
    print(json.dumps(OmegaConf.to_object(config), indent=2))
    print(f"Evaluation seed: {args.seed}")

    if args.batch_size is not None:
        print(f"Overriding batch size to {args.batch_size}")
        config.data.batch_size = args.batch_size

    model = AutoregressiveSampler.from_state(state)
    rng = jax.random.key(args.seed)
    rng_dataloader, rng_mc = jax.random.split(rng)
    dataloader = dataloader(rng_dataloader, config.data, config.kernel, args.N)

    evaluate(
        rng_mc,
        model,
        dataloader,
        strategies=["diagonal", "preserve", "ltr", "closest", "furthest", "random"],
        num_samples_for_random=10,
    )
