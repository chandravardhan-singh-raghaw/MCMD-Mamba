"""Training entrypoint: load config, build model/dataloaders, run training loop."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMD-Mamba training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()
    # TODO: load config (omegaconf/hydra), build model + datamodule, run engine
    print(f"Train with config={args.config} ckpt={args.ckpt_path} seed={args.seed}")


if __name__ == "__main__":
    main()
