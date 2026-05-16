import csv
import inspect
import os
import shutil
import sys

import torch
from monai.networks.nets import DynUNet, FlexibleUNet, SegResNet, SwinUNETR, UNETR, UNet, VNet
from monai.transforms import Compose, RandFlipd, RandGaussianNoised, RandRotate90d, RandSpatialCropd, RandZoomd, ToTensorD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ProteinLigandDatasetWithH5
from models.ConvNeXt3D import ConvNeXt3D
from models.ConvNeXt3DV2 import ConvNeXt3DV2
from models.LiteratureModels3D import (
    KalasantyUNet3D,
    PUResNetV1Like3D,
    PUResNetV2DenseLike3D,
    SwinSiteLike3D,
)
from models.ModernUNet3D import (
    CBAMUNet3D,
    CBAMResNet3D4LGN,
    ConvNeXtUNet3D,
    LightweightUNet3D,
    ResidualUNet3D,
    ResNet3D4LGN,
    SEResUNet3D,
    SEResNet3D4LGN,
    TinyConvNeXtUNet3D,
    UNetPlusPlus3D,
)
from models.ResNet3D4L import ResNet3D4L
from models.ResNet3D5L import ResNet3D5L
from models.ResNet3D6L import ResNet3D6L
from models.UNet3D4L import UNet3D4L
from models.UNet3D4LA import UNet3D4LA
from models.UNet3D4LAC import UNet3D4LAC
from models.UNet3D4LC import UNet3D4LC
from models.UNet3D4LStrided import UNet3D4LAStrided, UNet3D4LStrided
from models.UNet3D5L import UNet3D5L
from models.UNet3D6L import UNet3D6L
from transforms import CustomCompose, MonaiWrapper, RandomFlip, RandomRotate3D, Standardize
from utils.configuration import create_output_dirs, load_config, parse_args, setup_logger
from utils.pocket_metrics import (
    POCKET_PER_PROTEIN_FIELDNAMES,
    POCKET_SUMMARY_FIELDNAMES,
    POCKET_TOPK_COMPONENT_FIELDNAMES,
    POCKET_TOPK_PER_PROTEIN_FIELDNAMES,
    POCKET_TOPK_SUMMARY_FIELDNAMES,
    evaluate_pocket_metrics_for_sample,
    evaluate_topk_metrics_for_sample,
    normalize_postprocess_mode,
    select_best_paper_summary,
    summarize_pocket_metrics,
    summarize_topk_pocket_metrics,
)
from utils.training import (
    calculate_binary_stats_from_counts,
    calculate_binary_stats_from_probs,
    calculate_pos_weight_from_loader,
    get_device,
    get_loss_function,
    get_optimizer,
    get_scheduler,
    seed_worker,
    set_reproducibility,
)


MODEL_DICT = {
    "UNet3D4L": UNet3D4L,
    "UNet3D5L": UNet3D5L,
    "UNet3D6L": UNet3D6L,
    "UNet3D4LA": UNet3D4LA,
    "UNet3D4LC": UNet3D4LC,
    "UNet3D4LAC": UNet3D4LAC,
    "UNet3D4LStrided": UNet3D4LStrided,
    "UNet3D4LAStrided": UNet3D4LAStrided,
    "ResNet3D4L": ResNet3D4L,
    "ResNet3D5L": ResNet3D5L,
    "ResNet3D6L": ResNet3D6L,
    "PUResNetV1Like3D": PUResNetV1Like3D,
    "PUResNetV2DenseLike3D": PUResNetV2DenseLike3D,
    "KalasantyUNet3D": KalasantyUNet3D,
    "SwinSiteLike3D": SwinSiteLike3D,
    "MONAI_UNet3D": lambda in_ch, out_ch, base: UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=(base, base * 2, base * 4, base * 8, base * 16),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    ),
    "MONAI_DynUNet3D": lambda in_ch, out_ch, base: DynUNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=[3, 3, 3, 3],
        strides=[2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[base, base * 2, base * 4, base * 8, base * 16],
        dropout=0.0,
        norm_name="INSTANCE",
    ),
    "MONAI_FlexibleUNet3D": lambda in_ch, out_ch, base: FlexibleUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        backbone="resnet18",
        spatial_dims=3,
    ),
    "MONAI_UNETR": lambda in_ch, out_ch, base: UNETR(
        in_channels=in_ch,
        out_channels=out_ch,
        img_size=(128, 128, 128),
        feature_size=base,
    ),
    "MONAI_SwinUNETR": lambda in_ch, out_ch, base: SwinUNETR(
        in_chans=in_ch,
        out_chans=out_ch,
        img_size=(128, 128, 128),
        feature_size=base,
    ),
    "MONAI_VNet3D": lambda in_ch, out_ch, base: VNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        dropout_prob=0.0,
        act="elu",
    ),
    "MONAI_SegResNet3D": lambda in_ch, out_ch, base: SegResNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        init_filters=base,
    ),
    "ConvNeXt3D": lambda in_ch, out_ch, base: ConvNeXt3D(
        in_channels=in_ch,
        out_channels=out_ch,
        base_features=base,
        depths=[2, 2, 2, 2],
    ),
    "ConvNeXt3DV2": lambda in_ch, out_ch, base: ConvNeXt3DV2(
        in_channels=in_ch,
        out_channels=out_ch,
        base_features=base,
        depths=[2, 2, 2, 2],
    ),
    "ConvNeXtUNet3D": ConvNeXtUNet3D,
    "ResidualUNet3D": ResidualUNet3D,
    "SEResUNet3D": SEResUNet3D,
    "CBAMUNet3D": CBAMUNet3D,
    "LightweightUNet3D": LightweightUNet3D,
    "UNetPlusPlus3D": UNetPlusPlus3D,
    "ResNet3D4LGN": ResNet3D4LGN,
    "SEResNet3D4LGN": SEResNet3D4LGN,
    "CBAMResNet3D4LGN": CBAMResNet3D4LGN,
    "TinyConvNeXtUNet3D": TinyConvNeXtUNet3D,
}


def write_csv_header(path, fieldnames):
    with open(path, "w", newline="") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()


def append_csv_rows(path, fieldnames, rows):
    with open(path, "a", newline="") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writerows(rows)


def format_count(value):
    return f"{int(value):,}"


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def metric_postprocess_label(mode):
    if mode == "raw":
        return "Raw connected-component"
    if mode == "kalasanty_puresnet":
        return "Kalasanty/PUResNet-style"
    if mode == "puresnet_dbscan":
        return "PUResNet DBSCAN-style"
    return mode


def read_dataset_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_path(path, config_path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(config_path), path))


def resolve_dataset_lists(config, config_path):
    datasets = config.get("datasets", {})
    for split_name in ("train", "validation", "test"):
        file_key = f"{split_name}_file"
        if file_key in datasets:
            datasets[split_name] = read_dataset_list(resolve_path(datasets[file_key], config_path))
        elif isinstance(datasets.get(split_name), str):
            datasets[split_name] = read_dataset_list(resolve_path(datasets[split_name], config_path))
    config["datasets"] = datasets
    return config


def log_readable_validation_summary(
    logger,
    epoch,
    val_loss,
    threshold,
    primary_stats,
    best_threshold_stats,
    primary_paper_stats,
    best_paper_stats,
    dcc_cutoff_angstrom,
    dca_cutoff_angstrom,
    feature_set_label=None,
    model_label=None,
    comparison_paper_results=None,
):
    fixed_true_pocket_voxels = primary_stats["tp"] + primary_stats["fn"]
    fixed_predicted_pocket_voxels = primary_stats["tp"] + primary_stats["fp"]
    best_true_pocket_voxels = best_threshold_stats["tp"] + best_threshold_stats["fn"]
    best_predicted_pocket_voxels = best_threshold_stats["tp"] + best_threshold_stats["fp"]

    logger.info("Epoch %d validation summary", epoch)
    if model_label:
        logger.info("  Model: %s", model_label)
    if feature_set_label:
        logger.info("  Feature set: %s", feature_set_label)
    logger.info("  Loss: %.4f", val_loss)
    logger.info("")
    logger.info("  Fixed-threshold voxel mask score @ %.2f", threshold)
    logger.info(
        "    F1: %.4f | precision: %.4f | recall: %.4f",
        primary_stats["f1"],
        primary_stats["precision"],
        primary_stats["recall"],
    )
    logger.info(
        "    Predicted pocket voxels: %s | true pocket voxels: %s",
        format_count(fixed_predicted_pocket_voxels),
        format_count(fixed_true_pocket_voxels),
    )
    logger.info(
        "    Correct pocket voxels: %s | false pocket voxels: %s | missed pocket voxels: %s",
        format_count(primary_stats["tp"]),
        format_count(primary_stats["fp"]),
        format_count(primary_stats["fn"]),
    )
    if fixed_predicted_pocket_voxels == 0:
        logger.info(
            "    Interpretation: no voxel crossed threshold %.2f.",
            threshold,
        )

    logger.info("")
    logger.info("  Diagnostic best voxel threshold @ %.2f", best_threshold_stats["threshold"])
    logger.info(
        "    F1: %.4f | precision: %.4f | recall: %.4f",
        best_threshold_stats["f1"],
        best_threshold_stats["precision"],
        best_threshold_stats["recall"],
    )
    logger.info(
        "    Predicted pocket voxels: %s | true pocket voxels: %s",
        format_count(best_predicted_pocket_voxels),
        format_count(best_true_pocket_voxels),
    )
    logger.info(
        "    Correct pocket voxels: %s | false pocket voxels: %s | missed pocket voxels: %s",
        format_count(best_threshold_stats["tp"]),
        format_count(best_threshold_stats["fp"]),
        format_count(best_threshold_stats["fn"]),
    )
    if best_threshold_stats["recall"] > 0.90 and best_threshold_stats["precision"] < 0.05:
        logger.info(
            "    Interpretation: the model finds almost all true pocket voxels, but predicts far too much of the grid as pocket."
        )

    if primary_paper_stats is not None:
        logger.info("")
        logger.info("  Fixed-threshold pocket localization @ %.2f", threshold)
        logger.info(
            "    Center-distance success: %s/%s proteins within %.1f A",
            format_count(primary_paper_stats["dcc_success_count"]),
            format_count(primary_paper_stats["sample_count"]),
            dcc_cutoff_angstrom,
        )
        logger.info(
            "    No-prediction proteins: %s/%s",
            format_count(primary_paper_stats["no_prediction_count"]),
            format_count(primary_paper_stats["sample_count"]),
        )

    if best_paper_stats is not None:
        logger.info("")
        logger.info("  Diagnostic best pocket threshold @ %.2f", best_paper_stats["threshold"])
        logger.info(
            "    Center-distance success: %s/%s proteins within %.1f A",
            format_count(best_paper_stats["dcc_success_count"]),
            format_count(best_paper_stats["sample_count"]),
            dcc_cutoff_angstrom,
        )
        logger.info("    Pocket-level F1: %.4f", best_paper_stats["paper_f1"])
        logger.info(
            "    Average predicted pocket size: %s voxels",
            format_count(round(best_paper_stats["mean_predicted_positive_voxels"])),
        )
        logger.info(
            "    Selection score: %.4f",
            best_paper_stats["selection_score"],
        )

    for postprocess_mode, postprocess_result in (comparison_paper_results or {}).items():
        comparison_primary = postprocess_result.get("primary")
        comparison_best = postprocess_result.get("best")
        label = metric_postprocess_label(postprocess_mode)
        if comparison_primary is not None:
            logger.info("")
            logger.info("  %s fixed-threshold pocket localization @ %.2f", label, threshold)
            logger.info(
                "    Center-distance success: %s/%s proteins within %.1f A",
                format_count(comparison_primary["dcc_success_count"]),
                format_count(comparison_primary["sample_count"]),
                dcc_cutoff_angstrom,
            )
            logger.info(
                "    Pocket-level F1: %.4f | DCA@%.1fA: %.4f | DVO(success): %.4f",
                comparison_primary["paper_f1"],
                dca_cutoff_angstrom,
                comparison_primary["dca_success_rate_4a"],
                comparison_primary["mean_dvo_dcc_success"],
            )
            logger.info(
                "    No-prediction proteins: %s/%s",
                format_count(comparison_primary["no_prediction_count"]),
                format_count(comparison_primary["sample_count"]),
            )
        if comparison_best is not None:
            logger.info("")
            logger.info("  %s diagnostic best pocket threshold @ %.2f", label, comparison_best["threshold"])
            logger.info(
                "    Center-distance success: %s/%s proteins within %.1f A",
                format_count(comparison_best["dcc_success_count"]),
                format_count(comparison_best["sample_count"]),
                dcc_cutoff_angstrom,
            )
            logger.info("    Pocket-level F1: %.4f", comparison_best["paper_f1"])
            logger.info(
                "    Average predicted pocket size: %s voxels",
                format_count(round(comparison_best["mean_predicted_positive_voxels"])),
            )

def build_monai_transforms(config):
    augmentation = config.get("augmentation", {})
    transforms = []
    rotate_prob = float(augmentation.get("rotate90_prob", 0.5))
    flip_prob = float(augmentation.get("flip_axis_prob", 0.5))
    noise_prob = float(augmentation.get("gaussian_noise_prob", 0.1))
    zoom_prob = float(augmentation.get("zoom_prob", 0.2))

    if rotate_prob > 0:
        transforms.extend(
            [
                RandRotate90d(keys=["image", "label"], prob=rotate_prob, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image", "label"], prob=rotate_prob, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image", "label"], prob=rotate_prob, spatial_axes=(0, 2)),
            ]
        )
    if flip_prob > 0:
        transforms.extend(
            [
                RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),
            ]
        )
    if noise_prob > 0:
        transforms.append(RandGaussianNoised(keys=["image"], prob=noise_prob, mean=0.0, std=0.01))
    if zoom_prob > 0:
        transforms.append(RandZoomd(keys=["image", "label"], prob=zoom_prob, min_zoom=0.9, max_zoom=1.1))

    crop_size = augmentation.get("spatial_crop_size")
    if crop_size is None and config.get("use_monai_transforms", False):
        crop_size = (128, 128, 128)
    if crop_size:
        transforms.append(RandSpatialCropd(keys=["image", "label"], roi_size=tuple(crop_size), random_size=False))

    transforms.append(ToTensorD(keys=["image", "label"]))
    return Compose(transforms)


def build_transforms(config, training):
    augmentation = config.get("augmentation", {})
    standardize_enabled = bool(augmentation.get("standardize", True))
    channel_wise = bool(augmentation.get("standardize_channel_wise", True))
    transforms = []

    if training and bool(augmentation.get("enabled", True)):
        if config.get("use_monai_transforms", False):
            transforms.append(MonaiWrapper(build_monai_transforms(config)))
        else:
            transforms.extend(
                [
                    RandomFlip(axis_prob=float(augmentation.get("flip_axis_prob", 0.5))),
                    RandomRotate3D(prob=float(augmentation.get("rotate90_prob", 1.0))),
                ]
            )

    if standardize_enabled:
        transforms.append(Standardize(channel_wise=channel_wise))
    return CustomCompose(transforms)


def create_model(model_class, in_channels, base_features, model_dropout, device, logger):
    ModelClass = MODEL_DICT[model_class]
    if inspect.isclass(ModelClass):
        model_params = inspect.signature(ModelClass.__init__).parameters
        if "dropout" in model_params:
            return ModelClass(in_channels, 1, base_features, dropout=model_dropout).to(device)
        logger.warning("%s does not support dropout parameter, using model default", model_class)
        return ModelClass(in_channels, 1, base_features).to(device)
    return ModelClass(in_channels, 1, base_features).to(device)


def main():
    args = parse_args()

    config_path = args.config
    model_class = args.model
    base_features = args.base_features
    num_workers = args.num_workers
    base_model_output_dir = args.base_model_output_dir

    config = resolve_dataset_lists(load_config(config_path), config_path)
    validation_config = config.get("validation", {})
    seed = int(config.get("seed", config.get("training", {}).get("seed", 42)))
    set_reproducibility(seed)

    _, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)
    log_dir, weights_dir, tensorboard_dir = create_output_dirs(base_model_output_dir, config_name)
    run_dir = os.path.dirname(log_dir)
    shutil.copy2(config_path, os.path.join(run_dir, "config_snapshot.yml"))
    with open(os.path.join(run_dir, "run_command.txt"), "w") as command_file:
        command_file.write(" ".join(sys.argv) + "\n")

    logger = setup_logger(log_dir)
    num_epochs = int(config["training"]["num_epochs"])
    features = config["features"]
    feature_set = config.get("feature_set", {})
    feature_set_name = feature_set.get("name", config_name)
    feature_set_index = feature_set.get("index")
    feature_set_count = feature_set.get("count")
    if feature_set_index is not None and feature_set_count is not None:
        feature_set_label = f"{feature_set_index}/{feature_set_count} {feature_set_name}"
    else:
        feature_set_label = feature_set_name

    logger.info("---------------------------------")
    logger.info("Training is being started!")
    logger.info("---------------------------------")
    logger.info("Configuration name: %s", config_name)
    logger.info("Feature set: %s", feature_set_label)
    logger.info("Configuration file: %s", config_path)
    logger.info("Model: %s", model_class)
    logger.info("Base features: %d", base_features)
    logger.info("Seed: %d", seed)
    logger.info("Number of epochs: %d", num_epochs)
    logger.info("Features: %s", ", ".join(features))
    logger.info("Label: %s", config["label"])

    train_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"]["train"],
        transform=build_transforms(config, training=True),
        config_path=config_path,
    )
    validation_dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"].get("validation"),
        transform=build_transforms(config, training=False),
        config_path=config_path,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    worker_init_fn = seed_worker if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    device = get_device()
    logger.info("Device: %s", device)
    model = create_model(
        model_class=model_class,
        in_channels=len(features),
        base_features=base_features,
        model_dropout=config.get("model", {}).get("dropout", 0.5),
        device=device,
        logger=logger,
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(
        config["training"]["optimizer"],
        model.parameters(),
        config["training"]["learning_rate"],
        config["training"]["weight_decay"],
    )
    scheduler = get_scheduler(config["training"].get("scheduler"), optimizer)

    loss_config = config["training"].get("loss").copy()
    if loss_config.get("dynamic_pos_weight", True):
        max_batches = loss_config.get("pos_weight_max_batches", 50)
        calculated_pos_weight = calculate_pos_weight_from_loader(train_loader, max_batches=max_batches)
        loss_config["pos_weight"] = calculated_pos_weight
        logger.info("Using dynamically calculated pos_weight: %.1f", calculated_pos_weight)
    else:
        logger.info("Using fixed pos_weight from config: %s", loss_config.get("pos_weight", 1.0))
    criterion = get_loss_function(loss_config, device)

    writer = SummaryWriter(tensorboard_dir)

    threshold = float(validation_config.get("threshold", 0.5))
    threshold_sweep = validation_config.get(
        "threshold_sweep",
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    threshold_sweep = sorted({float(value) for value in threshold_sweep + [threshold, 0.4, 0.5]})

    paper_config = validation_config.get("paper_metrics", {})
    paper_metrics_enabled = bool(paper_config.get("enabled", True))
    dcc_cutoff_angstrom = float(paper_config.get("dcc_cutoff_angstrom", 4.0))
    dca_cutoff_angstrom = float(paper_config.get("dca_cutoff_angstrom", 4.0))
    min_component_voxels = int(paper_config.get("min_component_voxels", 5))
    min_component_volume_angstrom3 = paper_config.get("min_component_volume_angstrom3", 50.0)
    if min_component_volume_angstrom3 is not None:
        min_component_volume_angstrom3 = float(min_component_volume_angstrom3)
    primary_paper_postprocess = normalize_postprocess_mode(paper_config.get("postprocess", "raw"))
    comparison_postprocess_modes = [
        normalize_postprocess_mode(mode)
        for mode in as_list(paper_config.get("comparison_postprocess", ["kalasanty_puresnet"]))
    ]
    paper_postprocess_modes = []
    for mode in [primary_paper_postprocess, *comparison_postprocess_modes]:
        if mode not in paper_postprocess_modes:
            paper_postprocess_modes.append(mode)
    top_k_values = tuple(sorted({int(value) for value in paper_config.get("top_k", [1, 3])}))
    topk_metrics_enabled = bool(paper_config.get("top_k_metrics_enabled", True))
    topk_metric_values = tuple(
        sorted({1, 2, 3, *[int(value) for value in paper_config.get("top_k_metric_values", [1, 2, 3])]})
    )
    reference_pocket_count = int(paper_config.get("reference_pocket_count", 1))
    include_top_n_plus_2 = bool(paper_config.get("include_top_n_plus_2", True))
    selection_metric = paper_config.get("selection_metric", "paper_f1")
    selection_dvo_weight = float(paper_config.get("selection_dvo_weight", 5.0))
    selection_voxel_f1_weight = float(paper_config.get("selection_voxel_f1_weight", 1.0))
    selection_dca_weight = float(paper_config.get("selection_dca_weight", 0.25))
    selection_no_dcc_score_scale = float(paper_config.get("selection_no_dcc_score_scale", 0.05))
    selection_max_mean_predicted_positive_voxels = paper_config.get(
        "selection_max_mean_predicted_positive_voxels"
    )
    if selection_max_mean_predicted_positive_voxels is not None:
        selection_max_mean_predicted_positive_voxels = float(selection_max_mean_predicted_positive_voxels)

    logger.info("Validation voxel threshold: %.2f", threshold)
    logger.info("Validation threshold sweep: %s", threshold_sweep)
    if paper_metrics_enabled:
        logger.info(
            "Paper metrics enabled: DCC<=%.1fA, DCA<=%.1fA, min_component_voxels=%d, min_component_volume=%s A^3, top_k=%s",
            dcc_cutoff_angstrom,
            dca_cutoff_angstrom,
            min_component_voxels,
            min_component_volume_angstrom3,
            top_k_values,
        )
        if topk_metrics_enabled:
            logger.info(
                "Top-k metric CSVs enabled: top_k_metric_values=%s, reference_pocket_count=%d, include_top_n_plus_2=%s",
                topk_metric_values,
                reference_pocket_count,
                include_top_n_plus_2,
            )
        logger.info("Paper metric primary postprocess: %s", primary_paper_postprocess)
        logger.info(
            "Paper metric comparison postprocess: %s",
            [mode for mode in paper_postprocess_modes if mode != primary_paper_postprocess],
        )
        logger.info(
            "Paper model selection: metric=%s, dvo_weight=%.2f, voxel_f1_weight=%.2f, dca_weight=%.2f, no_dcc_score_scale=%.2f, max_mean_predicted_positive_voxels=%s",
            selection_metric,
            selection_dvo_weight,
            selection_voxel_f1_weight,
            selection_dca_weight,
            selection_no_dcc_score_scale,
            selection_max_mean_predicted_positive_voxels,
        )

    best_val_f1 = 0.0
    best_val_sweep_f1 = 0.0
    best_val_sweep_threshold = threshold
    best_val_selection_score = 0.0
    best_val_paper_f1 = 0.0
    best_val_paper_threshold = threshold
    best_val_paper_dcc_success = 0.0
    best_train_f1 = 0.0
    no_improvement_epochs = 0
    patience_config = config["training"].get("early_stopping_patience", 50)
    patience = None if patience_config is None else int(patience_config)
    if patience is not None and patience <= 0:
        patience = None

    total_batches_train = len(train_loader)
    total_batches_validation = len(validation_loader)

    threshold_sweep_path = os.path.join(log_dir, "validation_threshold_sweep.csv")
    best_thresholds_path = os.path.join(log_dir, "validation_best_thresholds.csv")
    per_protein_metrics_path = os.path.join(log_dir, "validation_per_protein_metrics.csv")
    paper_summary_path = os.path.join(log_dir, "validation_paper_metrics.csv")
    paper_per_protein_path = os.path.join(log_dir, "validation_paper_metrics_per_protein.csv")
    topk_summary_path = os.path.join(log_dir, "validation_paper_metrics_topk.csv")
    topk_per_protein_path = os.path.join(log_dir, "validation_paper_metrics_per_protein_topk.csv")
    topk_component_path = os.path.join(log_dir, "validation_paper_metrics_components_topk.csv")
    paper_summary_paths = {}
    paper_per_protein_paths = {}
    topk_summary_paths = {}
    topk_per_protein_paths = {}
    topk_component_paths = {}
    for postprocess_mode in paper_postprocess_modes:
        if postprocess_mode == primary_paper_postprocess:
            paper_summary_paths[postprocess_mode] = paper_summary_path
            paper_per_protein_paths[postprocess_mode] = paper_per_protein_path
            topk_summary_paths[postprocess_mode] = topk_summary_path
            topk_per_protein_paths[postprocess_mode] = topk_per_protein_path
            topk_component_paths[postprocess_mode] = topk_component_path
        else:
            paper_summary_paths[postprocess_mode] = os.path.join(
                log_dir,
                f"validation_paper_metrics_{postprocess_mode}.csv",
            )
            paper_per_protein_paths[postprocess_mode] = os.path.join(
                log_dir,
                f"validation_paper_metrics_per_protein_{postprocess_mode}.csv",
            )
            topk_summary_paths[postprocess_mode] = os.path.join(
                log_dir,
                f"validation_paper_metrics_topk_{postprocess_mode}.csv",
            )
            topk_per_protein_paths[postprocess_mode] = os.path.join(
                log_dir,
                f"validation_paper_metrics_per_protein_topk_{postprocess_mode}.csv",
            )
            topk_component_paths[postprocess_mode] = os.path.join(
                log_dir,
                f"validation_paper_metrics_components_topk_{postprocess_mode}.csv",
            )

    voxel_summary_fieldnames = ["epoch", "threshold", "f1", "precision", "recall", "tp", "fp", "tn", "fn"]
    best_threshold_fieldnames = [
        "epoch",
        "primary_threshold",
        "primary_voxel_f1",
        "primary_voxel_precision",
        "primary_voxel_recall",
        "primary_voxel_tp",
        "primary_voxel_fp",
        "primary_voxel_tn",
        "primary_voxel_fn",
        "best_voxel_threshold",
        "best_voxel_f1",
        "best_voxel_precision",
        "best_voxel_recall",
        "best_voxel_tp",
        "best_voxel_fp",
        "best_voxel_tn",
        "best_voxel_fn",
        "best_paper_threshold",
        "best_paper_f1",
        "best_paper_precision",
        "best_paper_recall",
        "best_paper_dcc_success_rate_4a",
        "best_paper_dca_success_rate_4a",
        "best_paper_dvo_all",
        "best_paper_dvo_dcc_success",
        "best_paper_dcc_success_count",
        "best_paper_mean_dcc_angstrom",
        "best_paper_mean_predicted_positive_voxels",
        "best_paper_no_prediction_count",
        "best_paper_selection_score",
        "paper_f1_fixed_threshold_040",
        "paper_f1_fixed_threshold_050",
    ]
    voxel_per_protein_fieldnames = [
        "epoch",
        "protein",
        "threshold",
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "tn",
        "fn",
        "positive_voxels",
        "predicted_positive_voxels",
    ]
    write_csv_header(threshold_sweep_path, voxel_summary_fieldnames)
    write_csv_header(best_thresholds_path, best_threshold_fieldnames)
    write_csv_header(per_protein_metrics_path, voxel_per_protein_fieldnames)
    if paper_metrics_enabled:
        for postprocess_mode in paper_postprocess_modes:
            write_csv_header(paper_summary_paths[postprocess_mode], POCKET_SUMMARY_FIELDNAMES)
            write_csv_header(paper_per_protein_paths[postprocess_mode], POCKET_PER_PROTEIN_FIELDNAMES)
            if topk_metrics_enabled:
                write_csv_header(topk_summary_paths[postprocess_mode], POCKET_TOPK_SUMMARY_FIELDNAMES)
                write_csv_header(topk_per_protein_paths[postprocess_mode], POCKET_TOPK_PER_PROTEIN_FIELDNAMES)
                write_csv_header(topk_component_paths[postprocess_mode], POCKET_TOPK_COMPONENT_FIELDNAMES)

    accumulation_steps = int(config["training"].get("accumulation_steps", 1))
    logger.info(
        "Using gradient accumulation with %d steps (effective batch size: %d)",
        accumulation_steps,
        config["training"]["batch_size"] * accumulation_steps,
    )

    for epoch in range(num_epochs):
        logger.info("")
        logger.info("%s", "=" * 60)
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)
        logger.info("%s", "=" * 60)
        logger.info("")
        logger.info("Training")

        model.train()
        train_loss = 0.0
        train_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        optimizer.zero_grad()

        for batch_idx, (protein, pocket_label) in enumerate(train_loader, start=1):
            protein, pocket_label = protein.to(device), pocket_label.to(device)
            output = model(protein).squeeze(1)
            loss = criterion(output, pocket_label)
            normalized_loss = loss / accumulation_steps
            normalized_loss.backward()

            if batch_idx % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            probabilities = torch.sigmoid(output).detach().cpu().numpy()
            targets = pocket_label.detach().cpu().numpy()
            train_batch_stats = calculate_binary_stats_from_probs(targets, probabilities, threshold)
            for key in train_counts:
                train_counts[key] += train_batch_stats[key]
            logger.info(
                "Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, Batch F1@%.2f: %.4f",
                epoch + 1,
                num_epochs,
                batch_idx,
                total_batches_train,
                loss.item(),
                threshold,
                train_batch_stats["f1"],
            )

        if total_batches_train % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.info("Applied final gradient accumulation step for remaining batches.")

        train_loss /= len(train_loader)
        train_stats = calculate_binary_stats_from_counts(
            train_counts["tp"],
            train_counts["fp"],
            train_counts["tn"],
            train_counts["fn"],
            threshold,
        )
        train_f1 = train_stats["f1"]
        train_precision = train_stats["precision"]
        train_recall = train_stats["recall"]

        logger.info(
            "Epoch %d Train Loss: %.4f, F1@%.2f: %.4f, Precision: %.4f, Recall: %.4f",
            epoch + 1,
            train_loss,
            threshold,
            train_f1,
            train_precision,
            train_recall,
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar("Recall/Train", train_recall, epoch)

        logger.info("")
        logger.info("Validation")

        model.eval()
        val_loss = 0.0
        primary_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        sweep_counts = {
            sweep_threshold: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for sweep_threshold in threshold_sweep
        }
        per_protein_rows = []
        paper_per_protein_rows_by_postprocess = {
            postprocess_mode: [] for postprocess_mode in paper_postprocess_modes
        }
        topk_per_protein_rows_by_postprocess = {
            postprocess_mode: [] for postprocess_mode in paper_postprocess_modes
        }
        topk_component_rows_by_postprocess = {
            postprocess_mode: [] for postprocess_mode in paper_postprocess_modes
        }

        with torch.no_grad():
            for batch_idx, (protein, pocket_label) in enumerate(validation_loader, start=1):
                protein, pocket_label = protein.to(device), pocket_label.to(device)
                output = model(protein).squeeze(1)
                val_loss += criterion(output, pocket_label).item()
                probabilities = torch.sigmoid(output).detach().cpu().numpy()
                targets = pocket_label.detach().cpu().numpy()
                primary_batch_stats = calculate_binary_stats_from_probs(targets, probabilities, threshold)

                for key in primary_counts:
                    primary_counts[key] += primary_batch_stats[key]

                for sweep_threshold in threshold_sweep:
                    sweep_batch_stats = calculate_binary_stats_from_probs(targets, probabilities, sweep_threshold)
                    for key in sweep_counts[sweep_threshold]:
                        sweep_counts[sweep_threshold][key] += sweep_batch_stats[key]

                batch_size = probabilities.shape[0]
                start_sample_idx = (batch_idx - 1) * validation_config["batch_size"]
                for sample_idx in range(batch_size):
                    dataset_idx = start_sample_idx + sample_idx
                    protein_name = validation_dataset.samples[dataset_idx][0]
                    protein_stats = calculate_binary_stats_from_probs(
                        targets[sample_idx],
                        probabilities[sample_idx],
                        threshold,
                    )
                    per_protein_rows.append({"epoch": epoch + 1, "protein": protein_name, **protein_stats})

                    if paper_metrics_enabled:
                        metadata = validation_dataset.get_metadata(dataset_idx)
                        ligand_mask = validation_dataset.load_metric_mask(dataset_idx, "features", "ligand")
                        for postprocess_mode in paper_postprocess_modes:
                            sample_paper_rows = evaluate_pocket_metrics_for_sample(
                                probabilities=probabilities[sample_idx],
                                label_mask=targets[sample_idx],
                                ligand_mask=ligand_mask,
                                protein_name=protein_name,
                                thresholds=threshold_sweep,
                                resolution=metadata["resolution"],
                                max_distance_angstrom=metadata["max_distance_angstrom"],
                                dcc_cutoff_angstrom=dcc_cutoff_angstrom,
                                dca_cutoff_angstrom=dca_cutoff_angstrom,
                                min_component_voxels=min_component_voxels,
                                min_component_volume_angstrom3=(
                                    min_component_volume_angstrom3
                                    if postprocess_mode != "raw"
                                    else None
                                ),
                                postprocess_mode=postprocess_mode,
                                top_k_values=top_k_values,
                            )
                            paper_per_protein_rows_by_postprocess[postprocess_mode].extend(
                                {"epoch": epoch + 1, **row} for row in sample_paper_rows
                            )
                            if topk_metrics_enabled:
                                component_rows, topk_rows = evaluate_topk_metrics_for_sample(
                                    probabilities=probabilities[sample_idx],
                                    label_mask=targets[sample_idx],
                                    ligand_mask=ligand_mask,
                                    protein_name=protein_name,
                                    thresholds=threshold_sweep,
                                    resolution=metadata["resolution"],
                                    max_distance_angstrom=metadata["max_distance_angstrom"],
                                    dcc_cutoff_angstrom=dcc_cutoff_angstrom,
                                    dca_cutoff_angstrom=dca_cutoff_angstrom,
                                    min_component_voxels=min_component_voxels,
                                    min_component_volume_angstrom3=(
                                        min_component_volume_angstrom3
                                        if postprocess_mode != "raw"
                                        else None
                                    ),
                                    postprocess_mode=postprocess_mode,
                                    top_k_values=topk_metric_values,
                                    reference_pocket_count=reference_pocket_count,
                                    include_top_n_plus_2=include_top_n_plus_2,
                                )
                                topk_context = {
                                    "epoch": epoch + 1,
                                    "run": config_name,
                                    "model": model_class,
                                    "feature_family": feature_set.get("feature_name", feature_set_name),
                                    "apbs_variant": feature_set.get("apbs_cutoff_variant", ""),
                                    "checkpoint": f"epoch_{epoch + 1}",
                                    "postprocess_mode": postprocess_mode,
                                }
                                topk_component_rows_by_postprocess[postprocess_mode].extend(
                                    {**topk_context, **row} for row in component_rows
                                )
                                topk_per_protein_rows_by_postprocess[postprocess_mode].extend(
                                    {**topk_context, **row} for row in topk_rows
                                )

                logger.info(
                    "Validation Iteration %d/%d, Batch F1@%.2f: %.4f",
                    batch_idx,
                    total_batches_validation,
                    threshold,
                    primary_batch_stats["f1"],
                )

        val_loss /= len(validation_loader)
        primary_stats = calculate_binary_stats_from_counts(
            primary_counts["tp"],
            primary_counts["fp"],
            primary_counts["tn"],
            primary_counts["fn"],
            threshold,
        )
        sweep_rows = [
            calculate_binary_stats_from_counts(
                counts["tp"],
                counts["fp"],
                counts["tn"],
                counts["fn"],
                sweep_threshold,
            )
            for sweep_threshold, counts in sweep_counts.items()
        ]
        sweep_rows_by_threshold = {row["threshold"]: row for row in sweep_rows}
        best_threshold_stats = max(sweep_rows, key=lambda row: row["f1"])
        val_f1 = primary_stats["f1"]
        val_precision = primary_stats["precision"]
        val_recall = primary_stats["recall"]
        best_val_f1 = max(best_val_f1, val_f1)
        if best_threshold_stats["f1"] > best_val_sweep_f1:
            best_val_sweep_f1 = best_threshold_stats["f1"]
            best_val_sweep_threshold = best_threshold_stats["threshold"]

        append_csv_rows(threshold_sweep_path, voxel_summary_fieldnames, [{"epoch": epoch + 1, **row} for row in sweep_rows])
        append_csv_rows(per_protein_metrics_path, voxel_per_protein_fieldnames, per_protein_rows)

        best_paper_stats = None
        primary_paper_stats = None
        primary_paper_summary_rows = []
        comparison_paper_results = {}
        if paper_metrics_enabled:
            for postprocess_mode, postprocess_rows in paper_per_protein_rows_by_postprocess.items():
                paper_summary_rows = summarize_pocket_metrics(
                    postprocess_rows,
                    threshold_sweep,
                    selection_metric=selection_metric,
                    selection_dvo_weight=selection_dvo_weight,
                    selection_voxel_f1_weight=selection_voxel_f1_weight,
                    selection_dca_weight=selection_dca_weight,
                    selection_no_dcc_score_scale=selection_no_dcc_score_scale,
                    selection_max_mean_predicted_positive_voxels=selection_max_mean_predicted_positive_voxels,
                    voxel_summary_by_threshold=sweep_rows_by_threshold,
                )
                postprocess_best_paper_stats = select_best_paper_summary(paper_summary_rows)
                postprocess_primary_paper_stats = next(
                    (row for row in paper_summary_rows if abs(row["threshold"] - threshold) < 1e-9),
                    None,
                )
                append_csv_rows(
                    paper_summary_paths[postprocess_mode],
                    POCKET_SUMMARY_FIELDNAMES,
                    [{"epoch": epoch + 1, **row} for row in paper_summary_rows],
                )
                append_csv_rows(
                    paper_per_protein_paths[postprocess_mode],
                    POCKET_PER_PROTEIN_FIELDNAMES,
                    postprocess_rows,
                )
                if topk_metrics_enabled:
                    topk_summary_rows = summarize_topk_pocket_metrics(
                        topk_per_protein_rows_by_postprocess[postprocess_mode]
                    )
                    append_csv_rows(
                        topk_summary_paths[postprocess_mode],
                        POCKET_TOPK_SUMMARY_FIELDNAMES,
                        topk_summary_rows,
                    )
                    append_csv_rows(
                        topk_per_protein_paths[postprocess_mode],
                        POCKET_TOPK_PER_PROTEIN_FIELDNAMES,
                        topk_per_protein_rows_by_postprocess[postprocess_mode],
                    )
                    append_csv_rows(
                        topk_component_paths[postprocess_mode],
                        POCKET_TOPK_COMPONENT_FIELDNAMES,
                        topk_component_rows_by_postprocess[postprocess_mode],
                    )

                if postprocess_mode == primary_paper_postprocess:
                    best_paper_stats = postprocess_best_paper_stats
                    primary_paper_stats = postprocess_primary_paper_stats
                    primary_paper_summary_rows = paper_summary_rows
                else:
                    comparison_paper_results[postprocess_mode] = {
                        "primary": postprocess_primary_paper_stats,
                        "best": postprocess_best_paper_stats,
                    }

        log_readable_validation_summary(
            logger=logger,
            epoch=epoch + 1,
            val_loss=val_loss,
            threshold=threshold,
            primary_stats=primary_stats,
            best_threshold_stats=best_threshold_stats,
            primary_paper_stats=primary_paper_stats,
            best_paper_stats=best_paper_stats,
            dcc_cutoff_angstrom=dcc_cutoff_angstrom,
            dca_cutoff_angstrom=dca_cutoff_angstrom,
            feature_set_label=feature_set_label,
            model_label=model_class,
            comparison_paper_results=comparison_paper_results,
        )

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("VoxelF1/Validation", val_f1, epoch)
        writer.add_scalar("VoxelF1/Validation_BestThreshold", best_threshold_stats["f1"], epoch)
        writer.add_scalar("Threshold/Validation_BestVoxelF1", best_threshold_stats["threshold"], epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)

        standard_score = best_threshold_stats["f1"]
        if paper_metrics_enabled and best_paper_stats is not None:
            standard_score = best_paper_stats["selection_score"]
            writer.add_scalar("PaperF1/Validation_BestThreshold", best_paper_stats["paper_f1"], epoch)
            writer.add_scalar("DCC4/Validation_BestThreshold", best_paper_stats["dcc_success_rate_4a"], epoch)
            writer.add_scalar("DCA4/Validation_BestThreshold", best_paper_stats["dca_success_rate_4a"], epoch)
            writer.add_scalar("DVO_All/Validation_BestThreshold", best_paper_stats["mean_dvo_all"], epoch)
            writer.add_scalar(
                "DVO_DCCSuccess/Validation_BestThreshold",
                best_paper_stats["mean_dvo_dcc_success"],
                epoch,
            )
            writer.add_scalar("Threshold/Validation_BestPaperF1", best_paper_stats["threshold"], epoch)
            for postprocess_mode, postprocess_result in comparison_paper_results.items():
                postprocess_best = postprocess_result.get("best")
                if postprocess_best is None:
                    continue
                scalar_prefix = f"PaperPostprocess/{postprocess_mode}"
                writer.add_scalar(f"{scalar_prefix}/PaperF1_BestThreshold", postprocess_best["paper_f1"], epoch)
                writer.add_scalar(
                    f"{scalar_prefix}/DCC4_BestThreshold",
                    postprocess_best["dcc_success_rate_4a"],
                    epoch,
                )
                writer.add_scalar(
                    f"{scalar_prefix}/DCA4_BestThreshold",
                    postprocess_best["dca_success_rate_4a"],
                    epoch,
                )
                writer.add_scalar(
                    f"{scalar_prefix}/DVO_DCCSuccess_BestThreshold",
                    postprocess_best["mean_dvo_dcc_success"],
                    epoch,
                )
                writer.add_scalar(f"{scalar_prefix}/BestThreshold", postprocess_best["threshold"], epoch)

        best_threshold_row = {
            "epoch": epoch + 1,
            "primary_threshold": threshold,
            "primary_voxel_f1": primary_stats["f1"],
            "primary_voxel_precision": primary_stats["precision"],
            "primary_voxel_recall": primary_stats["recall"],
            "primary_voxel_tp": primary_stats["tp"],
            "primary_voxel_fp": primary_stats["fp"],
            "primary_voxel_tn": primary_stats["tn"],
            "primary_voxel_fn": primary_stats["fn"],
            "best_voxel_threshold": best_threshold_stats["threshold"],
            "best_voxel_f1": best_threshold_stats["f1"],
            "best_voxel_precision": best_threshold_stats["precision"],
            "best_voxel_recall": best_threshold_stats["recall"],
            "best_voxel_tp": best_threshold_stats["tp"],
            "best_voxel_fp": best_threshold_stats["fp"],
            "best_voxel_tn": best_threshold_stats["tn"],
            "best_voxel_fn": best_threshold_stats["fn"],
            "best_paper_threshold": "",
            "best_paper_f1": "",
            "best_paper_precision": "",
            "best_paper_recall": "",
            "best_paper_dcc_success_rate_4a": "",
            "best_paper_dca_success_rate_4a": "",
            "best_paper_dvo_all": "",
            "best_paper_dvo_dcc_success": "",
            "best_paper_dcc_success_count": "",
            "best_paper_mean_dcc_angstrom": "",
            "best_paper_mean_predicted_positive_voxels": "",
            "best_paper_no_prediction_count": "",
            "best_paper_selection_score": "",
            "paper_f1_fixed_threshold_040": "",
            "paper_f1_fixed_threshold_050": "",
        }
        if best_paper_stats is not None:
            paper_stats_at_040 = next(
                (row for row in primary_paper_summary_rows if abs(row["threshold"] - 0.4) < 1e-9),
                None,
            )
            paper_stats_at_050 = next(
                (row for row in primary_paper_summary_rows if abs(row["threshold"] - 0.5) < 1e-9),
                None,
            )
            best_threshold_row.update(
                {
                    "best_paper_threshold": best_paper_stats["threshold"],
                    "best_paper_f1": best_paper_stats["paper_f1"],
                    "best_paper_precision": best_paper_stats["paper_precision"],
                    "best_paper_recall": best_paper_stats["paper_recall"],
                    "best_paper_dcc_success_rate_4a": best_paper_stats["dcc_success_rate_4a"],
                    "best_paper_dca_success_rate_4a": best_paper_stats["dca_success_rate_4a"],
                    "best_paper_dvo_all": best_paper_stats["mean_dvo_all"],
                    "best_paper_dvo_dcc_success": best_paper_stats["mean_dvo_dcc_success"],
                    "best_paper_dcc_success_count": best_paper_stats["dcc_success_count"],
                    "best_paper_mean_dcc_angstrom": best_paper_stats["mean_dcc_angstrom"],
                    "best_paper_mean_predicted_positive_voxels": best_paper_stats[
                        "mean_predicted_positive_voxels"
                    ],
                    "best_paper_no_prediction_count": best_paper_stats["no_prediction_count"],
                    "best_paper_selection_score": best_paper_stats["selection_score"],
                    "paper_f1_fixed_threshold_040": (
                        paper_stats_at_040["paper_f1"] if paper_stats_at_040 is not None else ""
                    ),
                    "paper_f1_fixed_threshold_050": (
                        paper_stats_at_050["paper_f1"] if paper_stats_at_050 is not None else ""
                    ),
                }
            )
        append_csv_rows(best_thresholds_path, best_threshold_fieldnames, [best_threshold_row])

        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, f"{model_class}_best_model_in_terms_of_training_score.pth"),
            )
            logger.info(
                "Checkpoint saved: new best training F1@%.2f: %.4f | epoch %d",
                threshold,
                best_train_f1,
                epoch + 1,
            )

        if standard_score > best_val_selection_score:
            best_val_selection_score = standard_score
            if best_paper_stats is not None:
                best_val_paper_f1 = best_paper_stats["paper_f1"]
                best_val_paper_threshold = best_paper_stats["threshold"]
                best_val_paper_dcc_success = best_paper_stats["dcc_success_rate_4a"]
            else:
                best_val_paper_f1 = standard_score
                best_val_paper_threshold = best_threshold_stats["threshold"]
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, f"{model_class}_best_model_in_terms_of_validation_paper_f1.pth"),
            )
            if best_paper_stats is not None:
                logger.info(
                    "Checkpoint saved: new best validation selection score %.4f | pocket F1 %.4f | threshold %.2f | epoch %d",
                    best_val_selection_score,
                    best_val_paper_f1,
                    best_val_paper_threshold,
                    epoch + 1,
                )
            else:
                logger.info(
                    "Checkpoint saved: new best validation selection score %.4f | voxel F1 %.4f | threshold %.2f | epoch %d",
                    best_val_selection_score,
                    best_val_paper_f1,
                    best_val_paper_threshold,
                    epoch + 1,
                )
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if patience is not None and no_improvement_epochs >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        if scheduler is not None:
            scheduler.step(val_loss)

    writer.close()
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
    if patience is None:
        logger.info("Early stopping: disabled")
    else:
        logger.info("Early stopping patience: %d", patience)
    if paper_metrics_enabled:
        logger.info("Primary paper postprocess: %s", primary_paper_postprocess)
        logger.info(
            "Comparison paper postprocess: %s",
            [mode for mode in paper_postprocess_modes if mode != primary_paper_postprocess],
        )
    logger.info("Best training voxel-F1@%.2f: %.4f", threshold, best_train_f1)
    logger.info("Best validation voxel-F1@%.2f: %.4f", threshold, best_val_f1)
    logger.info(
        "Best validation voxel threshold-sweep F1: %.4f at threshold %.2f",
        best_val_sweep_f1,
        best_val_sweep_threshold,
    )
    logger.info(
        "Best STANDARD validation selection score: %.4f, Pocket-F1: %.4f at threshold %.2f, DCC@4A: %.4f",
        best_val_selection_score,
        best_val_paper_f1,
        best_val_paper_threshold,
        best_val_paper_dcc_success,
    )
    logger.info("---------------------------------")


if __name__ == "__main__":
    main()
