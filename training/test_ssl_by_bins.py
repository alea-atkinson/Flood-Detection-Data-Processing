#!/usr/bin/env python3

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from your training script
from fine_tune_global_metrics import (
    FloodTileDataset,
    UNet,
    FocalDiceLoss,
    run_epoch,
)


def evaluate_flood_fraction(model, loader, device, threshold=0.5):
    """
    Computes the average actual and predicted flood fraction
    over all tiles in the dataset.
    """

    model.eval()

    gt_fractions = []
    pred_fractions = []

    with torch.no_grad():

        for images, masks, valid in loader:

            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs > threshold

            for i in range(images.shape[0]):

                valid_mask = valid[i] > 0.5

                gt_fraction = (
                    (masks[i] > 0.5)[valid_mask]
                    .float()
                    .mean()
                    .item()
                )

                pred_fraction = (
                    preds[i][valid_mask]
                    .float()
                    .mean()
                    .item()
                )

                gt_fractions.append(gt_fraction)
                pred_fractions.append(pred_fraction)

    return np.mean(gt_fractions), np.mean(pred_fractions)


def evaluate_precision_recall(model, loader, device, threshold=0.5):
    """
    Computes average precision and recall over all valid pixels.
    """

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for images, masks, valid in loader:

            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = probs > threshold

            targets = masks > 0.5
            valid_mask = valid > 0.5

            preds = preds & valid_mask
            targets = targets & valid_mask


            total_tp += (preds & targets).sum().item()
            total_fp += (preds & ~targets).sum().item()
            total_fn += (~preds & targets).sum().item()

    eps = 1e-7

    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)

    return precision, recall


def evaluate_prediction_confidence(model, loader, device, threshold=0.5):

    model.eval()

    correct_sum = 0.0
    incorrect_sum = 0.0

    correct_count = 0
    incorrect_count = 0

    with torch.no_grad():

        for images, masks, valid in loader:

            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)

            confidence = torch.abs(probs - 0.5) * 2 #normalize confidence 

            preds = probs > threshold
            targets = masks > 0.5
            valid_mask = valid > 0.5

            correct = (preds == targets) & valid_mask
            incorrect = (preds != targets) & valid_mask

            correct_sum += confidence[correct].sum().item()
            incorrect_sum += confidence[incorrect].sum().item()

            correct_count += correct.sum().item()
            incorrect_count += incorrect.sum().item()

    eps = 1e-7

    mean_correct = correct_sum / (correct_count + eps)
    mean_incorrect = incorrect_sum / (incorrect_count + eps)

    return mean_correct, mean_incorrect


def parse_args():

    fp = 7

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file to evaluate",
        default=f"training/fine_tune_csvs/florence_lofpo_flood_bin_tests/fp{fp}",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Saved model (.pt)", 
        default= f"models/ssl_no_loca_florence_lopfo_milton_no_overlap_fp{fp}_best.pt",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=2,
        help="Tile index to visualize",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    targets = [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 100]


    for lower, upper in zip(targets[:-1], targets[1:]):
        test_csv = Path(f"{args.csv}/{lower}-{upper}.csv")

        ####################################
        # Load dataset
        ####################################

        dataset = FloodTileDataset(test_csv)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        ####################################
        # Load model
        ####################################

        checkpoint = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False,
        )

        model = UNet(
            in_channels=3,
            out_channels=1,
            base_channels=checkpoint["args"]["base_channels"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        ####################################
        # Evaluate
        ####################################

        loss_fn = FocalDiceLoss()

        metrics = run_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
        )

        precision, recall = evaluate_precision_recall(
            model,
            loader,
            device,
        )

        avg_gt_fraction, avg_pred_fraction = evaluate_flood_fraction(
            model,
            loader,
            device,
        )

        mean_correct, mean_incorrect = evaluate_prediction_confidence(
            model,
            loader,
            device,
        )

        print("\nEvaluation Results")
        print("------------------")
        print(f"Dataset : {test_csv}")
        print(f"Model: {args.checkpoint}")
        print(f"Tiles : {len(dataset)}")
        print(f"Loss  : {metrics['loss']:.4f}")
        print(f"Dice  : {metrics['dice']:.4f}")
        print(f"IoU   : {metrics['iou']:.4f}")
        print(f"Precision   : {metrics['precision']:.4f}")
        print(f"Recall      : {metrics['recall']:.4f}")
        print(f"Average actual flood fraction    : {avg_gt_fraction:.4%}")
        print(f"Average predicted flood fraction : {avg_pred_fraction:.4%}")
        print(f"Mean confidence (correct predictions)   : {mean_correct:.4f}")
        print(f"Mean confidence (incorrect predictions) : {mean_incorrect:.4f}")



if __name__ == "__main__":
    main()