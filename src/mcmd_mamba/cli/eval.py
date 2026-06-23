"""Evaluation entrypoint: load checkpoint, run on test set, report metrics."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMD-Mamba evaluation")
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (optional)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save-confusion", type=str, default=None, help="Save confusion matrix path")
    args = parser.parse_args()
    # TODO: load model from ckpt, run evaluator, print/save metrics
    print(f"Eval ckpt={args.ckpt_path} split={args.split} save_confusion={args.save_confusion}")


if __name__ == "__main__":
    main()
