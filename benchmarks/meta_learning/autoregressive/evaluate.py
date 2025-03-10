import csv
import os
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
    N: int,  # len of the dataloader
    strategies: list[Strategy],
    num_samples_for_random: int,
    batching_for_random: int | None,
):
    nlls = defaultdict(list)
    max_nll = defaultdict(lambda: float("-inf"))

    logfile = open(Path(os.environ["RESULTS_DIR"]) / "log.csv", "a", newline="")
    writer = csv.DictWriter(logfile, strategies)
    writer.writeheader()

    pbar = tqdm(dataloader, total=N, desc="Evaluating")

    for i, datum in enumerate(pbar):
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
                batching_for_random,
            )
            max = nll.max()
            if max_nll[strategy] < max:
                print(
                    f"New lowest probability assigned for {strategy}: batch {i}, index in batch {nll.argmax()}, nll {max}"
                )
                max_nll[strategy] = max
            nll = np.mean(nll)  # average nll over batch
            nlls[strategy].append(nll)
        writer.writerow({strategy: nll[-1] for strategy, nll in nlls.items()})
        pbar.set_postfix(
            {f"NLL {strategy}": np.mean(nlls[strategy]) for strategy in strategies}
        )


# TODO @pgrynfelder: add other dataloaders
def dataloader(rng, data, kernel, N):
    gp_dataloader = build_gp_dataloader(data, kernel)(rng)
    num_ctx = data.num_ctx.max
    for i, datum in enumerate(gp_dataloader):
        if i >= N:
            raise StopIteration

        s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, var, ls, period = datum

        # note that s, f come directly from the GP not the observation process
        # TODO @pgrynfelder: might be desirable to change this
        yield s_ctx, f_ctx, s[:, num_ctx:], f[:, num_ctx:], valid_lens_ctx


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-N", type=int, default=250, help="evaluate against N*batch_size sample paths"
    )
    parser.add_argument(
        "-M",
        type=int,
        default=100,
        help="num samples for the SMC estimate of logpdf for the random strategy",
    )
    parser.add_argument(
        "-Mb",
        type=int,
        default=None,
        help="how to batch the SMC (batch size effectively becomes batch_size*this)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the batch size set in config (recommended 128)",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))

    args = parser.parse_args()

    os.environ["RESULTS_DIR"] = str(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    state, config = load_ckpt(args.model)
    print(json.dumps(OmegaConf.to_object(config), indent=2))
    print(f"Evaluation seed: {args.seed}")

    if args.batch_size is not None:
        print(f"Overriding batch size to {args.batch_size}")
        config.data.batch_size = args.batch_size
    print(f"SMC estimate for random logpdf using {args.M} samples.")

    model = AutoregressiveSampler.from_state(state)
    rng = jax.random.key(args.seed)
    rng_dataloader, rng_mc = jax.random.split(rng)
    dataloader = dataloader(rng_dataloader, config.data, config.kernel, args.N)

    evaluate(
        rng_mc,
        model,
        dataloader,
        args.N,
        strategies=["diagonal", "ltr", "closest", "furthest", "random"],
        num_samples_for_random=args.M,
        batching_for_random=args.Mb,
    )
