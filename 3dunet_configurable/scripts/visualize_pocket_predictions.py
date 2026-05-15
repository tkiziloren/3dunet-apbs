import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import normalize_feature
from main import create_model
from transforms import Standardize
from utils.pocket_metrics import center_of_mask, extract_components

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


SUMMARY_FIELDNAMES = [
    "protein",
    "threshold",
    "voxel_f1",
    "voxel_precision",
    "voxel_recall",
    "tp",
    "fp",
    "fn",
    "label_voxels",
    "predicted_positive_voxels",
    "top_component_voxels",
    "dcc_to_ligand_center_angstrom",
    "dcc_to_label_center_angstrom",
    "dca_to_ligand_atoms_angstrom",
    "dvo_top_component",
    "label_center_zyx",
    "ligand_center_zyx",
    "prediction_center_zyx",
    "figure_path",
]


class NullLogger:
    def warning(self, *args, **kwargs):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize pocket predictions for selected validation proteins.")
    parser.add_argument("--config", required=True, help="Training config used for the run.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--model", default="UNet3D4LStrided", help="Model class name.")
    parser.add_argument("--base-features", type=int, default=8, help="Base feature count used by the checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures and summary CSV will be written.")
    parser.add_argument("--proteins", nargs="+", required=True, help="Protein IDs to visualize.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6], help="Prediction thresholds.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device.")
    parser.add_argument("--max-points", type=int, default=2500, help="Maximum points per scatter layer.")
    return parser.parse_args()


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_state_dict(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def load_sample(h5_path, features, label_name, standardize_enabled=True, channel_wise=False):
    with h5py.File(h5_path, "r") as h5f:
        feature_arrays = []
        for feature_name in features:
            if "features" in h5f and feature_name in h5f["features"]:
                array = h5f["features"][feature_name][:]
            elif feature_name in h5f:
                array = h5f[feature_name][:]
            else:
                raise KeyError(f"Feature '{feature_name}' not found in {h5_path}")
            feature_arrays.append(normalize_feature(array, feature_name))

        if "label" in h5f and label_name in h5f["label"]:
            label = h5f["label"][label_name][:]
        elif label_name in h5f:
            label = h5f[label_name][:]
        else:
            raise KeyError(f"Label '{label_name}' not found in {h5_path}")

        ligand = h5f["features"]["ligand"][:] if "features" in h5f and "ligand" in h5f["features"] else None
        resolution = float(h5f.attrs.get("resolution", 1.0))

    protein = torch.tensor(np.stack(feature_arrays), dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    if standardize_enabled:
        protein, label_tensor = Standardize(channel_wise=channel_wise)(protein, label_tensor)

    return protein, label.astype(np.uint8), None if ligand is None else ligand.astype(np.uint8), resolution


def binary_stats(label_mask, probabilities, threshold):
    label_bool = label_mask > 0.5
    pred_bool = probabilities > threshold
    tp = int(np.logical_and(pred_bool, label_bool).sum())
    fp = int(np.logical_and(pred_bool, ~label_bool).sum())
    fn = int(np.logical_and(~pred_bool, label_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def distance_angstrom(center_a, center_b, resolution):
    if center_a is None or center_b is None:
        return float("inf")
    return float(np.linalg.norm((np.asarray(center_a) - np.asarray(center_b)) * resolution))


def dca_angstrom(pred_center, ligand_coords, resolution):
    if pred_center is None or ligand_coords is None or len(ligand_coords) == 0:
        return float("inf")
    distances = np.linalg.norm((ligand_coords - np.asarray(pred_center)) * resolution, axis=1)
    return float(distances.min())


def dvo(component_mask, label_mask):
    if component_mask is None:
        return 0.0
    label_bool = label_mask > 0.5
    union = np.logical_or(component_mask, label_bool).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(component_mask, label_bool).sum() / union)


def format_center(center):
    if center is None:
        return ""
    return ",".join(f"{value:.2f}" for value in center)


def zyx_to_xyz(coords):
    coords = np.asarray(coords)
    if coords.ndim == 1:
        return np.asarray([coords[2], coords[1], coords[0]])
    return coords[:, [2, 1, 0]]


def downsample(coords, max_points):
    if len(coords) <= max_points:
        return coords
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return coords[indices]


def add_scatter(ax, coords, color, label, marker="o", alpha=0.35, size=8, max_points=2500):
    if coords is None or len(coords) == 0:
        return
    coords = downsample(np.asarray(coords), max_points)
    xyz = zyx_to_xyz(coords)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, label=label, marker=marker, alpha=alpha, s=size)


def add_center(ax, center, color, label, marker, size=90):
    if center is None:
        return
    xyz = zyx_to_xyz(np.asarray(center))
    ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], c=color, label=label, marker=marker, s=size, edgecolors="black")


def projection_rgb(label_mask, component_mask, ligand_mask, axis):
    label_proj = np.max(label_mask > 0.5, axis=axis)
    pred_proj = np.zeros_like(label_proj, dtype=bool) if component_mask is None else np.max(component_mask, axis=axis)
    ligand_proj = np.zeros_like(label_proj, dtype=bool) if ligand_mask is None else np.max(ligand_mask > 0.5, axis=axis)

    rgb = np.zeros((*label_proj.shape, 3), dtype=np.float32)
    rgb[..., 0] = pred_proj.astype(np.float32)
    rgb[..., 1] = label_proj.astype(np.float32)
    rgb[..., 2] = ligand_proj.astype(np.float32)
    overlap = pred_proj & label_proj
    rgb[..., 0][overlap] = 1.0
    rgb[..., 1][overlap] = 1.0
    return rgb


def plot_sample(
    output_path,
    protein,
    threshold,
    probabilities,
    label_mask,
    ligand_mask,
    component_mask,
    metrics,
    centers,
    max_points,
):
    label_center, ligand_center, pred_center = centers
    label_coords = np.argwhere(label_mask > 0.5)
    ligand_coords = None if ligand_mask is None else np.argwhere(ligand_mask > 0.5)
    component_coords = None if component_mask is None else np.argwhere(component_mask)

    fig = plt.figure(figsize=(15, 10))
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    add_scatter(ax_3d, label_coords, "#2ca02c", "cavity6 label", alpha=0.35, size=12, max_points=max_points)
    add_scatter(ax_3d, ligand_coords, "#1f77b4", "ligand", alpha=0.45, size=12, max_points=max_points)
    add_scatter(ax_3d, component_coords, "#d62728", "top prediction", alpha=0.45, size=13, max_points=max_points)
    add_center(ax_3d, label_center, "#2ca02c", "label center", "*")
    add_center(ax_3d, ligand_center, "#1f77b4", "ligand center", "D")
    add_center(ax_3d, pred_center, "#d62728", "prediction center", "X")
    if pred_center is not None and ligand_center is not None:
        line = np.vstack([zyx_to_xyz(pred_center), zyx_to_xyz(ligand_center)])
        ax_3d.plot(line[:, 0], line[:, 1], line[:, 2], color="#444444", linewidth=1.2)
    ax_3d.set_title("3D voxel view")
    ax_3d.set_xlabel("X voxel")
    ax_3d.set_ylabel("Y voxel")
    ax_3d.set_zlabel("Z voxel")
    ax_3d.set_xlim(0, label_mask.shape[2])
    ax_3d.set_ylim(0, label_mask.shape[1])
    ax_3d.set_zlim(0, label_mask.shape[0])
    ax_3d.legend(loc="upper left", fontsize=8)

    axes = [fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3), fig.add_subplot(2, 3, 5)]
    projection_specs = [(0, "Z max projection"), (1, "Y max projection"), (2, "X max projection")]
    for ax, (axis, title) in zip(axes, projection_specs):
        ax.imshow(projection_rgb(label_mask, component_mask, ligand_mask, axis), origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_text = fig.add_subplot(2, 3, 6)
    ax_text.axis("off")
    text = "\n".join(
        [
            f"Protein: {protein}",
            f"Threshold: {threshold:.2f}",
            f"Voxel F1: {metrics['voxel_f1']:.4f}",
            f"Precision / recall: {metrics['precision']:.4f} / {metrics['recall']:.4f}",
            f"TP / FP / FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
            f"Predicted positives: {metrics['predicted_positive_voxels']}",
            f"Top component voxels: {metrics['top_component_voxels']}",
            f"DCC to ligand center: {metrics['dcc_to_ligand_center_angstrom']:.2f} A",
            f"DCC to label center: {metrics['dcc_to_label_center_angstrom']:.2f} A",
            f"DCA to ligand atoms: {metrics['dca_to_ligand_atoms_angstrom']:.2f} A",
            f"DVO top component: {metrics['dvo_top_component']:.4f}",
            "",
            "Colors:",
            "red = top predicted component",
            "green = cavity6 label",
            "blue = ligand",
            "yellow = prediction-label overlap",
        ]
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle(f"{protein} pocket diagnostic", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    features = config["features"]
    model = create_model(
        model_class=args.model,
        in_channels=len(features),
        base_features=args.base_features,
        model_dropout=config.get("model", {}).get("dropout", 0.5),
        device=device,
        logger=NullLogger(),
    )
    load_state_dict(model, args.checkpoint, device)
    model.eval()

    augmentation = config.get("augmentation", {})
    standardize_enabled = bool(augmentation.get("standardize", True))
    channel_wise = bool(augmentation.get("standardize_channel_wise", True))
    h5_dir = Path(config["h5_directory"])
    summary_rows = []

    for protein in args.proteins:
        h5_path = h5_dir / f"{protein}.h5"
        protein_tensor, label_mask, ligand_mask, resolution = load_sample(
            h5_path,
            features,
            config["label"],
            standardize_enabled=standardize_enabled,
            channel_wise=channel_wise,
        )
        with torch.no_grad():
            logits = model(protein_tensor.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        label_center, _ = center_of_mask(label_mask)
        ligand_center, ligand_coords = center_of_mask(ligand_mask) if ligand_mask is not None else (None, None)

        for threshold in args.thresholds:
            stats = binary_stats(label_mask, probabilities, threshold)
            components, _, predicted_positive_voxels, labeled_components = extract_components(
                probabilities,
                threshold,
                min_component_voxels=int(config["validation"]["paper_metrics"].get("min_component_voxels", 5)),
                max_components=3,
            )
            top_component = components[0] if components else None
            component_mask = labeled_components == top_component["component_id"] if top_component else None
            pred_center = top_component["center"] if top_component else None
            top_component_voxels = int(top_component["voxel_count"]) if top_component else 0

            metrics = {
                "voxel_f1": stats["f1"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "predicted_positive_voxels": int(predicted_positive_voxels),
                "top_component_voxels": top_component_voxels,
                "dcc_to_ligand_center_angstrom": distance_angstrom(pred_center, ligand_center, resolution),
                "dcc_to_label_center_angstrom": distance_angstrom(pred_center, label_center, resolution),
                "dca_to_ligand_atoms_angstrom": dca_angstrom(pred_center, ligand_coords, resolution),
                "dvo_top_component": dvo(component_mask, label_mask),
            }

            figure_path = output_dir / f"{protein}_threshold_{threshold:.2f}.png"
            plot_sample(
                figure_path,
                protein,
                threshold,
                probabilities,
                label_mask,
                ligand_mask,
                component_mask,
                metrics,
                (label_center, ligand_center, pred_center),
                args.max_points,
            )
            summary_rows.append(
                {
                    "protein": protein,
                    "threshold": f"{threshold:.2f}",
                    "voxel_f1": metrics["voxel_f1"],
                    "voxel_precision": metrics["precision"],
                    "voxel_recall": metrics["recall"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "label_voxels": int((label_mask > 0.5).sum()),
                    "predicted_positive_voxels": metrics["predicted_positive_voxels"],
                    "top_component_voxels": metrics["top_component_voxels"],
                    "dcc_to_ligand_center_angstrom": metrics["dcc_to_ligand_center_angstrom"],
                    "dcc_to_label_center_angstrom": metrics["dcc_to_label_center_angstrom"],
                    "dca_to_ligand_atoms_angstrom": metrics["dca_to_ligand_atoms_angstrom"],
                    "dvo_top_component": metrics["dvo_top_component"],
                    "label_center_zyx": format_center(label_center),
                    "ligand_center_zyx": format_center(ligand_center),
                    "prediction_center_zyx": format_center(pred_center),
                    "figure_path": str(figure_path),
                }
            )

    summary_path = output_dir / "visual_diagnostics_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} diagnostics to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
