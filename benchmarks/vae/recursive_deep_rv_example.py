import argparse

from dl4bi.vae import RecursiveGMLPDeepRV
from dl4bi.vae.train_utils import (
    recursive_deep_rv_train_step,
    recursive_deep_rv_valid_step,
)

from deep_rv_example import run_example


def main(seed=57, gt_ls=20, num_cycles=5):
    run_example(
        seed=seed,
        gt_ls=gt_ls,
        nn_model=RecursiveGMLPDeepRV(num_cycles=num_cycles),
        train_step=recursive_deep_rv_train_step,
        valid_step_fn=recursive_deep_rv_valid_step,
        model_label=f"Recursive DeepRV ({num_cycles} cycles)",
        decode_fn=lambda x: x[-1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--gt-ls", type=float, default=20.0)
    parser.add_argument("--num-cycles", type=int, default=5)
    args = parser.parse_args()
    main(args.seed, args.gt_ls, args.num_cycles)
