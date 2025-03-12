import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Generator

import jax
import numpy as np
from dataloader import build_gp_dataloader
from omegaconf import OmegaConf
from tqdm import tqdm

from dl4bi.meta_learning.autoregressive import AutoregressiveSampler, Strategy
from dl4bi.meta_learning.train_utils import load_ckpt


def evaluate(
    rng,
    model: AutoregressiveSampler,
    dataloader: Generator,
    N: int,  # how many samples to take from the dataloader
    strategies: list[Strategy],
    num_samples_for_random: int,
    batching_for_random: int | None,
    save_full_log_every: int | None,
):
    nlls = defaultdict(list)

    results_dir = Path(os.environ["RESULTS_DIR"])
    logfile = open(results_dir / "log.csv", "a", newline="")

    pbar = tqdm(dataloader, total=N, desc="Evaluating")

    for i, datum in enumerate(pbar):
        if i >= N:
            break
        (s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, f_test_obs) = datum
        for strategy in strategies:
            rng, rng_i = jax.random.split(rng)
            nll_obs = -model.logpdf(
                rng_i,
                s_ctx,
                f_ctx,
                s_test,
                f_test_obs,
                valid_lens_ctx,
                strategy,
                num_samples_for_random,
                batching_for_random,
            )
            nll_gp = -model.logpdf(
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

            # save data
            nlls[f"nll_obs_{strategy}"].append(nll_obs)
            nlls[f"nll_gp_{strategy}"].append(nll_gp)

        # log batch-mean nll to csv
        if i == 0:
            writer = csv.DictWriter(logfile, list(nlls.keys()))
            writer.writeheader()
        writer.writerow({strategy: np.mean(nll[-1]) for strategy, nll in nlls.items()})
        # report running mean to tqdm
        pbar.set_postfix({strategy: np.mean(nll) for strategy, nll in nlls.items()})

        if save_full_log_every and i % save_full_log_every == 0:
            np.savez(
                results_dir / "full_log.npz",
                **{strategy: np.array(nll) for strategy, nll in nlls.items()},
            )

    np.savez(
        results_dir / "full_log.npz",
        **{strategy: np.array(nll) for strategy, nll in nlls.items()},
    )


# TODO @pgrynfelder: add other dataloaders
def dataloader(rng, data, kernel):
    gp_dataloader = build_gp_dataloader(data, kernel)(rng)
    num_ctx = data.num_ctx.max
    for datum in gp_dataloader:
        (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            s_test,
            f_test,
            f_test_obs,
            valid_lens_test,
            var,
            ls,
            period,
        ) = datum

        # note that s, f come directly from the GP not the observation process
        # TODO @pgrynfelder: might be desirable to change this
        yield (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            s_test[:, num_ctx:],
            f_test[:, num_ctx:],
            f_test_obs[:, num_ctx:],
        )


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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / f"autoregressive {datetime.now()}",
    )

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
    dataloader = dataloader(rng_dataloader, config.data, config.kernel)

    evaluate(
        rng_mc,
        model,
        dataloader,
        args.N,
        strategies=["diagonal", "ltr", "closest", "furthest", "random"],
        num_samples_for_random=args.M,
        batching_for_random=args.Mb,
        save_full_log_every=10,
    )
