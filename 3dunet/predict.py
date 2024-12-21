import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ProteinLigandDatasetWithH5
from model import UNet3D
from transforms import Standardize, CustomCompose
from utils.configuration import setup_logger, load_config
from utils.training import get_device, calculate_metrics, get_loss_function, initialize_metrics


def main(config_path):
    # Load Config and Initialize Logger
    config = load_config(config_path)
    config_dir, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)
    base_output_dir = os.path.join("output", "test_results", config_name)
    os.makedirs(base_output_dir, exist_ok=True)

    log_dir = os.path.join(base_output_dir, "logs")
    logger = setup_logger(log_dir)
    logger.info("Test process started!")

    # Load Model
    device = get_device()
    model = UNet3D().to(device)
    trained_model_path = os.path.join("output", config_name, "best_model.pth")
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    logger.info(f"Model loaded from {trained_model_path}")

    # Test Dataset and DataLoader
    test_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"]["test"],
        transform=CustomCompose([Standardize()])
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Loss Function and Metrics
    criterion = get_loss_function(config["training"].get("loss"), device)
    torch_metrics = initialize_metrics(threshold=0.5, device=device)

    # Testing Process
    test_loss = 0
    test_f1_sum, test_precision_sum, test_recall_sum = 0, 0, 0
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0

    output_dir = os.path.join(base_output_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (protein, pocket_label) in enumerate(test_loader, start=1):
            protein, pocket_label = protein.to(device), pocket_label.to(device)
            output = model(protein).squeeze(1)
            loss = criterion(output, pocket_label)
            test_loss += loss.item()

            # Metrics Calculation
            f1, precision, recall, (tp, fp, tn, fn) = calculate_metrics(pocket_label, output, torch_metrics)
            test_f1_sum += f1
            test_precision_sum += precision
            test_recall_sum += recall
            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn

            # Save Predictions and Labels
            output_probs = torch.sigmoid(output).cpu().numpy()
            np.save(os.path.join(output_dir, f"pred_batch_{batch_idx}.npy"), output_probs)
            np.save(os.path.join(output_dir, f"true_batch_{batch_idx}.npy"), pocket_label.cpu().numpy())

            # Visualization for first 5 samples
            if batch_idx <= 5:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Protein Input (Slice)")
                plt.imshow(protein[0, 0, :, :, 80].cpu().numpy(), cmap="viridis")
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.title("True Pocket Label (Slice)")
                plt.imshow(pocket_label[0, :, :, 80].cpu().numpy(), cmap="coolwarm")
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.title("Predicted Pocket (Slice)")
                plt.imshow(output_probs[0, :, :, 80], cmap="coolwarm")
                plt.colorbar()

                plt.savefig(os.path.join(output_dir, f"visual_batch_{batch_idx}.png"))
                plt.close()

    # Calculate Mean Metrics
    test_loss /= len(test_loader)
    test_f1 = test_f1_sum / len(test_loader)
    test_precision = test_precision_sum / len(test_loader)
    test_recall = test_recall_sum / len(test_loader)

    logger.info(f"Test Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    logger.info(f"Confusion Matrix - TP: {all_tp}, FP: {all_fp}, TN: {all_tn}, FN: {all_fn}")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Confusion Matrix - TP: {all_tp}, FP: {all_fp}, TN: {all_tn}, FN: {all_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained UNet3D model.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)


