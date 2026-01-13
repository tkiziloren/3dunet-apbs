import os

import torch
from monai.networks.nets import UNet, DynUNet, SegResNet, UNETR, SwinUNETR, FlexibleUNet, VNet
from monai.transforms import Compose, RandRotate90d, RandFlipd, RandGaussianNoised, RandZoomd, RandSpatialCropd, ToTensorD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ProteinLigandDatasetWithH5
from models.UNet3D4L import UNet3D4L
from models.UNet3D5L import UNet3D5L
from models.UNet3D6L import UNet3D6L
from models.UNet3D4LA import UNet3D4LA
from models.UNet3D4LC import UNet3D4LC
from models.UNet3D4LAC import UNet3D4LAC
from models.ResNet3D4L import ResNet3D4L
from models.ResNet3D5L import ResNet3D5L
from models.ResNet3D6L import ResNet3D6L
from models.ConvNeXt3D import ConvNeXt3D
from models.ConvNeXt3DV2 import ConvNeXt3DV2
from transforms import RandomFlip, RandomRotate3D, Standardize, CustomCompose, MonaiWrapper
from monai.transforms import (
    RandRotate90,
    RandFlip,
    RandGaussianNoise,
    RandZoom,
    RandSpatialCrop,
    Compose
)
from utils.configuration import setup_logger, parse_args, load_config, create_output_dirs
from utils.training import get_optimizer, get_scheduler, get_loss_function, get_device, initialize_metrics, calculate_metrics


train_transforms = Compose([
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1, 2]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
    RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1),
    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
    ToTensorD(keys=["image", "label"])
])

MODEL_DICT = {
                "UNet3D4L": UNet3D4L,
                "UNet3D5L": UNet3D5L,
                "UNet3D6L": UNet3D6L,
                "UNet3D4LA": UNet3D4LA,
                "UNet3D4LC": UNet3D4LC,
                "UNet3D4LAC": UNet3D4LAC,
                "ResNet3D4L": ResNet3D4L,
                "ResNet3D5L": ResNet3D5L,
                "ResNet3D6L": ResNet3D6L,
                # MONAI'nin klasik 3D UNet'i
                "MONAI_UNet3D": lambda in_ch, out_ch, base:
                UNet(
                        spatial_dims=3,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        channels=(base, base * 2, base * 4, base * 8, base * 16),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                        norm='batch'
                    ),
                    # MONAI'nin dynamic UNet'i
                    "MONAI_DynUNet3D": lambda in_ch, out_ch, base:
                    DynUNet(
                        spatial_dims=3,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=[3, 3, 3, 3],
                        strides=[2, 2, 2, 2],
                        upsample_kernel_size=[2, 2, 2, 2],
                        filters=[base, base * 2, base * 4, base * 8, base * 16],
                        dropout=0.0,
                        norm_name="INSTANCE"
                    ),
                    # Backbone'ı seçilebilir UNet
                    "MONAI_FlexibleUNet3D": lambda in_ch, out_ch, base:
                    FlexibleUNet(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        backbone="resnet18",  # eg: "resnet34"
                        spatial_dims=3,
                    ),

                    "MONAI_UNETR": lambda in_ch, out_ch, base:
                    UNETR(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        img_size=(128, 128, 128),
                        feature_size=base
                    ),

                    # Swin Transformer tabanlı UNet (daha çok RAM!)
                    "MONAI_SwinUNETR": lambda in_ch, out_ch, base:
                    SwinUNETR(
                        in_chans=in_ch,
                        out_chans=out_ch,
                        img_size=(128, 128, 128),
                        feature_size=base
                    ),

                    # VNet (segmentasyon için)
                    "MONAI_VNet3D": lambda in_ch, out_ch, base:
                    VNet(
                        spatial_dims=3,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout_prob=0.0,
                        act="elu"
                    ),

                    # ResNet tabanlı segmentasyon modeli
                    "MONAI_SegResNet3D": lambda in_ch, out_ch, base:
                    SegResNet(
                        spatial_dims=3,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        init_filters=base
                    ),

                    # Kendi ConvNeXt3D implementasyonun (segmentasyon için)
                    "ConvNeXt3D": lambda in_ch, out_ch, base:
                    ConvNeXt3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        base_features=base,
                        depths=[2, 2, 2, 2]
                    ),
                    "ConvNeXt3DV2": lambda in_ch, out_ch, base:
                    ConvNeXt3DV2(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        base_features=base,
                        depths=[2, 2, 2, 2]
                    ),
                }




if __name__ == "__main__":

    args = parse_args()

    config_path = args.config
    model_class = args.model
    base_features = args.base_features
    num_workers = args.num_workers
    base_model_output_dir = args.base_model_output_dir

    config = load_config(config_path)
    config_dir, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)  # "codon/pdbbind.yml" -> "pdbbind"
    log_dir, weights_dir, tensorboard_dir = create_output_dirs(base_model_output_dir, config_name)
    logger = setup_logger(log_dir)
    num_epochs = config["training"]["num_epochs"]

    features = config["features"]
    label_name = config["label"]

    logger.info("---------------------------------")
    logger.info("Training is being started!")
    logger.info("---------------------------------")
    logger.info("Training parameters:")
    logger.info("Configuration name: %s", config_name)
    logger.info("Configuration file: %s", config_path)
    logger.info("Model: %s", model_class)
    logger.info("Base features: %d", base_features)
    logger.info("Number of epochs: %d", num_epochs)
    logger.info("Model class: %s", model_class)

    monai_transforms = Compose([
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1),
        RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
        ToTensorD(keys=["image", "label"])
    ])

    transform_default = CustomCompose([RandomFlip(), RandomRotate3D(), Standardize()])
    transform_monai = CustomCompose([MonaiWrapper(monai_transforms), Standardize()])

    train_transform = transform_monai if config.get("use_monai_transforms", False) else transform_default
    # Dataset ve DataLoader
    train_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"]["train"],
        transform=train_transform,
        config_path=config_path
    )
    validation_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"].get("validation"),
        transform=CustomCompose([Standardize()]),
        config_path=config_path
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=config["validation"]["batch_size"], shuffle=False, num_workers=num_workers)

    # Model, optimizer, scheduler, loss
    device = get_device()
    ModelClass = MODEL_DICT[model_class]
    #model = ModelClass(in_channels=2, out_channels=1, base_features=base_features).to(device)
    model = ModelClass(len(features), 1, base_features).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(config["training"]["optimizer"], model.parameters(), config["training"]["learning_rate"], config["training"]["weight_decay"])
    scheduler = get_scheduler(config["training"].get("scheduler"), optimizer)
    criterion = get_loss_function(config["training"].get("loss"), device)
    writer = SummaryWriter(tensorboard_dir)

    # Training loop
    best_val_f1 = 0.0
    best_train_f1 = 0.0
    no_improvement_epochs = 0
    threshold = 0.5
    patience = config["training"].get("early_stopping_patience", 50)

    total_batches_train = len(train_loader)
    total_batches_validation = len(validation_loader)
    torch_metrics = initialize_metrics(threshold=threshold, device=device)
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

        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_class}_best_model_in_terms_of_training_score.pth"))
            logger.info(f"New best model saved for training score: {best_train_f1} at epoch {epoch + 1}")
        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_class}_best_model_in_terms_of_validation_score.pth"))
            logger.info(f"New best model saved for validation: {best_val_f1} score at epoch {epoch + 1}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        scheduler.step(val_loss)

    writer.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_class}_final_model.pth"))
    logger.info("Final model saved.")

    logger.info("Training completed.")
    logger.info("---------------------------------")
    logger.info("Summary of training:")
    logger.info("---------------------------------")
    logger.info("Configuration name: %s", config_name)
    logger.info("Configuration file: %s", config_path)
    logger.info("Model: %s", model_class)
    logger.info("Base features: %d", base_features)
    logger.info("Number of epochs: %d", num_epochs)
    logger.info("Batch size: %d", config["training"]["batch_size"])
    logger.info("Learning rate: %f", config["training"]["learning_rate"])
    logger.info("Weight decay: %f", config["training"]["weight_decay"])
    logger.info("Optimizer: %s", config["training"]["optimizer"])
    logger.info("Scheduler: %s", config["training"].get("scheduler"))
    logger.info("Loss function: %s", config["training"].get("loss"))
    logger.info("Early stopping patience: %d", patience)
    logger.info(f"Best validation F1 score: {best_val_f1:.4f}, best training f1 score: {best_train_f1:.4f}")
    logger.info("---------------------------------")