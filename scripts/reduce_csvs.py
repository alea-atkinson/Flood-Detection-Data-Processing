#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import argparse


def reduce_csv(input_csv, output_csv, fraction=0.1, seed=42):
    """
    Randomly keeps a fraction of rows from a CSV file.

    Args:
        input_csv: Path to input CSV
        output_csv: Path to save reduced CSV
        fraction: Fraction of rows to keep
        seed: Random seed for reproducibility
    """

    df = pd.read_csv(input_csv)

    original_size = len(df)

    # Randomly sample rows
    df_reduced = df.sample(
        frac=fraction,
        random_state=seed
    ).reset_index(drop=True)

    df_reduced.to_csv(output_csv, index=False)

    print(f"{input_csv}")
    print(f"  Original rows: {original_size}")
    print(f"  New rows:      {len(df_reduced)}")
    print(f"  Saved to:      {output_csv}")
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Randomly reduce training and validation CSV files"
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default = "training/new_csvs/florence_lopfo/heldout_fp1/train.csv",
        help="Path to training CSV"
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        default = "training/new_csvs/florence_lopfo/heldout_fp1/val.csv",
        help="Path to validation CSV"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/fine_tune_csvs/reduced_csvs/new",
        help="Directory to save reduced CSV files"
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of rows to keep"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    reduce_csv(
        args.train_csv,
        output_dir / "train_ten_fp1.csv",
        args.fraction,
        args.seed
    )

    reduce_csv(
        args.val_csv,
        output_dir / "val_ten_fp1.csv",
        args.fraction,
        args.seed
    )
