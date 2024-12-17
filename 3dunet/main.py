import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ProteinLigandDatasetWithH5
from model import UNet3D
from transforms import RandomFlip, RandomRotate3D, Standardize, CustomCompose
from utils.configuration import setup_logger, parse_args, load_config, create_output_dirs
from utils.training import get_optimizer, get_scheduler, get_loss_function, get_device, initialize_metrics, calculate_metrics

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    config_dir, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)  # "codon/pdbbind.yml" -> "pdbbind"
    base_output_dir = os.path.join("output", os.path.basename(config_dir))
    log_dir, weights_dir, tensorboard_dir = create_output_dirs(base_output_dir, config_name)
    logger = setup_logger(log_dir)
    logger.info("Training started!")

    # Dataset ve DataLoader
    train_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"]["train"],
        transform=CustomCompose([RandomFlip(), RandomRotate3D(), Standardize()])
    )
    validation_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"].get("validation"),
        transform=CustomCompose([Standardize()])
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=config["validation"]["batch_size"], shuffle=False, num_workers=8)

    # Model, optimizer, scheduler, loss
    device = get_device()
    model = UNet3D().to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(config["training"]["optimizer"], model.parameters(), config["training"]["learning_rate"], config["training"]["weight_decay"])
    scheduler = get_scheduler(config["training"].get("scheduler"), optimizer)
    criterion = get_loss_function(config["training"].get("loss"), device)
    writer = SummaryWriter(tensorboard_dir)

    # Training loop
    best_val_f1 = 0.0
    no_improvement_epochs = 0
    threshold = 0.5
    patience = config["training"].get("early_stopping_patience", 10)

    total_batches_train = len(train_loader)
    total_batches_validation = len(validation_loader)
    num_epochs = config["training"]["num_epochs"]
    torch_metrics = initialize_metrics(threshold=0.5, device=device)
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        train_loss, train_f1_sum, train_precision_sum, train_recall_sum = 0, 0, 0, 0
        for batch_idx, (protein, pocket_label) in enumerate(train_loader, start=1):
            protein, pocket_label = protein.to(device), pocket_label.to(device)
            optimizer.zero_grad()
            output = model(protein).squeeze(1)
            loss = criterion(output, pocket_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            f1, precision, recall, _ = calculate_metrics(pocket_label, output, torch_metrics)
            train_f1_sum += f1
            train_precision_sum += precision
            train_recall_sum += recall
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{batch_idx}/{total_batches_train}], Loss: {loss.item():.4f}, Batch F1: {f1:.4f}")

        train_loss /= len(train_loader)
        train_f1 = train_f1_sum / len(train_loader)
        train_precision = train_precision_sum / len(train_loader)
        train_recall = train_recall_sum / len(train_loader)

        logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar("Recall/Train", train_recall, epoch)

        # Validation
        model.eval()
        val_loss, val_f1_sum, val_precision_sum, val_recall_sum = 0, 0, 0, 0
        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0

        with torch.no_grad():
            for batch_idx, (protein, pocket_label) in enumerate(validation_loader, start=1):
                protein, pocket_label = protein.to(device), pocket_label.to(device)
                output = model(protein).squeeze(1)
                val_loss += criterion(output, pocket_label).item()
                f1, precision, recall, (tp, fp, tn, fn) = calculate_metrics(pocket_label, output, torch_metrics)
                val_f1_sum += f1
                val_precision_sum += precision
                val_recall_sum += recall
                all_tp += tp
                all_fp += fp
                all_tn += tn
                all_fn += fn
                logger.info(f"Validation Iteration {batch_idx}/{total_batches_validation}, Batch F1: {f1:.4f}")

        val_loss /= len(validation_loader)
        val_f1 = val_f1_sum / len(validation_loader)
        val_precision = val_precision_sum / len(validation_loader)
        val_recall = val_recall_sum / len(validation_loader)

        logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        logger.info(f"Confusion Matrix - TP: {all_tp}, FP: {all_fp}, TN: {all_tn}, FN: {all_fn}")
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)

        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(weights_dir, 'best_model.pth'))
            logger.info(f"New best model saved at epoch {epoch + 1}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        scheduler.step(val_loss)

    writer.close()
    logger.info("Training completed.")
