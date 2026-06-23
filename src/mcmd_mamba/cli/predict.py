"""Prediction entrypoint: load model, run on images, output class/logits."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMD-Mamba prediction")
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Image path or directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV/path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (optional)")
    args = parser.parse_args()
    # TODO: load model, run inference, write predictions
    print(f"Predict ckpt={args.ckpt_path} input={args.input} output={args.output}")


if __name__ == "__main__":
    main()
