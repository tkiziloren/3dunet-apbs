import math
from collections import defaultdict

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    find_objects,
    generate_binary_structure,
    label as connected_components,
)
from scipy.spatial import cKDTree


RAW_POSTPROCESS = "raw"
KALASANTY_PURESNET_POSTPROCESS = "kalasanty_puresnet"
PURESNET_DBSCAN_POSTPROCESS = "puresnet_dbscan"
SUPPORTED_POSTPROCESS_MODES = {
    RAW_POSTPROCESS,
    KALASANTY_PURESNET_POSTPROCESS,
    PURESNET_DBSCAN_POSTPROCESS,
}

PURESNET_DBSCAN_EPS_ANGSTROM = 5.5
DBSCAN_EXACT_MAX_POINTS = 100_000


POCKET_SUMMARY_FIELDNAMES = [
    "epoch",
    "threshold",
    "paper_f1",
    "paper_precision",
    "paper_recall",
    "paper_tp",
    "paper_fp",
    "paper_fn",
    "strict_f1",
    "strict_precision",
    "strict_recall",
    "strict_tp",
    "strict_fp",
    "strict_fn",
    "dcc_success_rate_4a",
    "dca_success_rate_4a",
    "top1_dcc_success_rate_4a",
    "top3_dcc_success_rate_4a",
    "mean_dcc_angstrom",
    "median_dcc_angstrom",
    "mean_dcc_to_label_angstrom",
    "mean_dca_angstrom",
    "mean_dvo_all",
    "mean_dvo_dcc_success",
    "mean_pli_all",
    "mean_pli_dcc_success",
    "dcc_success_count",
    "mean_predicted_positive_voxels",
    "mean_top_component_voxels",
    "no_prediction_count",
    "sample_count",
    "selection_metric",
    "selection_score",
]


POCKET_PER_PROTEIN_FIELDNAMES = [
    "epoch",
    "protein",
    "threshold",
    "predicted",
    "component_count",
    "top_component_voxels",
    "predicted_positive_voxels",
    "dcc_angstrom",
    "dcc_to_label_angstrom",
    "dcc_to_ligand_angstrom",
    "dca_angstrom",
    "dvo",
    "pli",
    "dcc_success_4a",
    "dca_success_4a",
    "top1_dcc_success_4a",
    "top3_dcc_success_4a",
    "paper_tp",
    "paper_fp",
    "paper_fn",
    "strict_tp",
    "strict_fp",
    "strict_fn",
    "dcc_reference",
]


POCKET_TOPK_COMPONENT_FIELDNAMES = [
    "epoch",
    "run",
    "model",
    "feature_family",
    "apbs_variant",
    "checkpoint",
    "postprocess_mode",
    "protein",
    "threshold",
    "component_rank",
    "component_id",
    "component_voxels",
    "score_sum",
    "score_mean",
    "center_x",
    "center_y",
    "center_z",
    "dcc_angstrom",
    "dcc_to_label_angstrom",
    "dcc_to_ligand_angstrom",
    "dca_angstrom",
    "dvo",
    "pli",
    "dcc_success_4a",
    "dca_success_4a",
    "dcc_reference",
]


POCKET_TOPK_PER_PROTEIN_FIELDNAMES = [
    "epoch",
    "run",
    "model",
    "feature_family",
    "apbs_variant",
    "checkpoint",
    "postprocess_mode",
    "protein",
    "threshold",
    "top_k_label",
    "top_k",
    "reference_pocket_count",
    "predicted",
    "candidate_count",
    "component_count",
    "predicted_positive_voxels",
    "best_dcc_angstrom",
    "best_dcc_component_rank",
    "dvo_of_best_dcc_component",
    "pli_of_best_dcc_component",
    "best_dca_angstrom",
    "best_dca_component_rank",
    "dvo_of_best_dca_component",
    "pli_of_best_dca_component",
    "best_dvo",
    "best_dvo_component_rank",
    "dcc_of_best_dvo_component",
    "dca_of_best_dvo_component",
    "pli_of_best_dvo_component",
    "best_pli",
    "best_pli_component_rank",
    "dcc_of_best_pli_component",
    "dca_of_best_pli_component",
    "dvo_of_best_pli_component",
    "best_dvo_dcc_success_4a",
    "best_dvo_dca_success_4a",
    "best_pli_dcc_success_4a",
    "best_pli_dca_success_4a",
    "dcc_success_4a",
    "dca_success_4a",
    "paper_tp",
    "paper_fp",
    "paper_fn",
    "dcc_reference",
]


POCKET_TOPK_SUMMARY_FIELDNAMES = [
    "epoch",
    "run",
    "model",
    "feature_family",
    "apbs_variant",
    "checkpoint",
    "postprocess_mode",
    "threshold",
    "top_k_label",
    "top_k",
    "pocket_f1",
    "pocket_precision",
    "pocket_recall",
    "paper_tp",
    "paper_fp",
    "paper_fn",
    "dcc_success_rate_4a",
    "dca_success_rate_4a",
    "mean_best_dcc_angstrom",
    "mean_best_dca_angstrom",
    "mean_best_dvo_all",
    "mean_best_pli_all",
    "mean_dcc_of_best_dvo_angstrom",
    "mean_dca_of_best_dvo_angstrom",
    "mean_dcc_of_best_pli_angstrom",
    "mean_dca_of_best_pli_angstrom",
    "best_dvo_dcc_success_rate_4a",
    "best_dvo_dca_success_rate_4a",
    "best_pli_dcc_success_rate_4a",
    "best_pli_dca_success_rate_4a",
    "mean_dvo_of_best_dcc_all",
    "mean_dvo_of_best_dcc_success",
    "mean_pli_of_best_dcc_all",
    "mean_pli_of_best_dcc_success",
    "mean_best_pli_dcc_success",
    "mean_predicted_positive_voxels",
    "mean_candidate_count",
    "no_prediction_count",
    "sample_count",
]


def calculate_f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def center_of_mask(mask):
    coords = np.argwhere(np.asarray(mask) > 0.5)
    if coords.size == 0:
        return None, coords
    return coords.mean(axis=0), coords


def normalize_postprocess_mode(postprocess_mode):
    aliases = {
        "none": RAW_POSTPROCESS,
        "connected_components": RAW_POSTPROCESS,
        "kalasanty": KALASANTY_PURESNET_POSTPROCESS,
        "puresnet": KALASANTY_PURESNET_POSTPROCESS,
        "puresnet_v1": KALASANTY_PURESNET_POSTPROCESS,
        "kalasanty_puresnet_v1": KALASANTY_PURESNET_POSTPROCESS,
        "dbscan": PURESNET_DBSCAN_POSTPROCESS,
        "puresnet_dbscan": PURESNET_DBSCAN_POSTPROCESS,
        "puresnet_v2": PURESNET_DBSCAN_POSTPROCESS,
        "puresnetv2": PURESNET_DBSCAN_POSTPROCESS,
        "puresnet_v2_dbscan": PURESNET_DBSCAN_POSTPROCESS,
    }
    mode = aliases.get(str(postprocess_mode).strip().lower(), str(postprocess_mode).strip().lower())
    if mode not in SUPPORTED_POSTPROCESS_MODES:
        raise ValueError(
            f"Unsupported pocket postprocess mode '{postprocess_mode}'. "
            f"Supported modes: {sorted(SUPPORTED_POSTPROCESS_MODES)}"
        )
    return mode


def _clear_border(mask, structure):
    labeled, _ = connected_components(mask, structure=structure)
    if labeled.max() == 0:
        return mask

    border_labels = set()
    for axis in range(mask.ndim):
        low_slice = [slice(None)] * mask.ndim
        high_slice = [slice(None)] * mask.ndim
        low_slice[axis] = 0
        high_slice[axis] = -1
        border_labels.update(np.unique(labeled[tuple(low_slice)]))
        border_labels.update(np.unique(labeled[tuple(high_slice)]))

    border_labels.discard(0)
    if not border_labels:
        return mask
    return mask & ~np.isin(labeled, list(border_labels))


def _minimum_component_voxels(min_component_voxels, min_component_volume_angstrom3, resolution):
    min_voxels = int(min_component_voxels)
    if min_component_volume_angstrom3 is None:
        return min_voxels
    voxel_volume = float(resolution) ** 3
    if voxel_volume <= 0:
        return min_voxels
    return max(min_voxels, int(math.ceil(float(min_component_volume_angstrom3) / voxel_volume)))


def _postprocess_mask(
    pred_mask,
    postprocess_mode,
    structure,
):
    if postprocess_mode == RAW_POSTPROCESS:
        return pred_mask
    if postprocess_mode == KALASANTY_PURESNET_POSTPROCESS:
        closed = binary_closing(pred_mask, structure=structure)
        return _clear_border(closed, structure=structure)
    if postprocess_mode == PURESNET_DBSCAN_POSTPROCESS:
        return pred_mask
    raise ValueError(f"Unsupported pocket postprocess mode: {postprocess_mode}")


def _ball_structure(radius_voxels):
    radius_voxels = max(1, int(radius_voxels))
    coords = np.indices((2 * radius_voxels + 1,) * 3, dtype=np.float32)
    center = float(radius_voxels)
    squared_distance = sum((coords[axis] - center) ** 2 for axis in range(3))
    return squared_distance <= float(radius_voxels) ** 2


def _dbscan_labels_exact(coords, shape, eps_voxels):
    tree = cKDTree(coords.astype(np.float32, copy=False))
    visited = np.zeros(coords.shape[0], dtype=bool)
    point_labels = np.zeros(coords.shape[0], dtype=np.int32)
    cluster_id = 0

    for point_idx in range(coords.shape[0]):
        if visited[point_idx]:
            continue
        cluster_id += 1
        queue = [point_idx]
        visited[point_idx] = True
        point_labels[point_idx] = cluster_id
        queue_pos = 0

        while queue_pos < len(queue):
            current_idx = queue[queue_pos]
            queue_pos += 1
            neighbor_indices = tree.query_ball_point(coords[current_idx], eps_voxels)
            for neighbor_idx in neighbor_indices:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    point_labels[neighbor_idx] = cluster_id
                    queue.append(neighbor_idx)
                elif point_labels[neighbor_idx] == 0:
                    point_labels[neighbor_idx] = cluster_id

    labeled = np.zeros(shape, dtype=np.int32)
    labeled[tuple(coords.T)] = point_labels
    return labeled, int(cluster_id)


def _dbscan_labels_approximate(pred_mask, eps_voxels):
    radius = max(1, int(math.ceil(eps_voxels / 2.0)))
    expanded = binary_dilation(pred_mask, structure=_ball_structure(radius))
    expanded_labels, component_count = connected_components(expanded)
    labeled = np.where(pred_mask, expanded_labels, 0).astype(np.int32, copy=False)
    return labeled, int(component_count)


def _puresnet_dbscan_labels(pred_mask, resolution):
    coords = np.argwhere(pred_mask)
    if coords.size == 0:
        return np.zeros(pred_mask.shape, dtype=np.int32), 0

    eps_voxels = PURESNET_DBSCAN_EPS_ANGSTROM / max(float(resolution), 1e-6)
    if coords.shape[0] <= DBSCAN_EXACT_MAX_POINTS:
        return _dbscan_labels_exact(coords, pred_mask.shape, eps_voxels)
    return _dbscan_labels_approximate(pred_mask, eps_voxels)


def _components_from_labeled(
    probabilities,
    labeled,
    component_count,
    predicted_positive_voxels,
    minimum_voxels,
    max_components,
):
    if predicted_positive_voxels == 0 or component_count == 0:
        return [], int(component_count), int(predicted_positive_voxels), labeled

    component_slices = find_objects(labeled)
    components = []

    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_labels = labeled[component_slice]
        local_mask = local_labels == component_id
        voxel_count = int(local_mask.sum())
        if voxel_count < minimum_voxels:
            continue

        local_probabilities = probabilities[component_slice]
        score_sum = float(local_probabilities[local_mask].sum())
        score_mean = float(local_probabilities[local_mask].mean()) if voxel_count else 0.0
        local_coords = np.argwhere(local_mask)
        offset = np.asarray([axis_slice.start for axis_slice in component_slice], dtype=np.float64)
        center = local_coords.mean(axis=0) + offset
        components.append(
            {
                "component_id": component_id,
                "voxel_count": voxel_count,
                "score_sum": score_sum,
                "score_mean": score_mean,
                "center": center,
            }
        )

    components.sort(key=lambda item: (item["score_sum"], item["voxel_count"]), reverse=True)
    return components[:max_components], int(component_count), int(predicted_positive_voxels), labeled


def extract_components(
    probabilities,
    threshold,
    min_component_voxels=5,
    max_components=3,
    postprocess_mode=RAW_POSTPROCESS,
    resolution=1.0,
    min_component_volume_angstrom3=None,
):
    postprocess_mode = normalize_postprocess_mode(postprocess_mode)
    pred_mask = np.asarray(probabilities) > threshold
    structure = generate_binary_structure(rank=3, connectivity=3)
    pred_mask = _postprocess_mask(pred_mask, postprocess_mode, structure=structure)
    minimum_voxels = _minimum_component_voxels(
        min_component_voxels,
        min_component_volume_angstrom3,
        resolution,
    )
    predicted_positive_voxels = int(pred_mask.sum())
    if predicted_positive_voxels == 0:
        return [], 0, predicted_positive_voxels, None

    if postprocess_mode == PURESNET_DBSCAN_POSTPROCESS:
        labeled, component_count = _puresnet_dbscan_labels(pred_mask, resolution)
    else:
        labeled, component_count = connected_components(pred_mask, structure=structure)
    return _components_from_labeled(
        probabilities,
        labeled,
        component_count,
        predicted_positive_voxels,
        minimum_voxels,
        max_components,
    )


def _distance_angstrom(center_a, center_b, resolution):
    if center_a is None or center_b is None:
        return math.inf
    return float(np.linalg.norm((np.asarray(center_a) - np.asarray(center_b)) * resolution))


def _dca_angstrom(predicted_center, ligand_coords, resolution):
    if predicted_center is None or ligand_coords is None or len(ligand_coords) == 0:
        return math.inf
    distances = np.linalg.norm((ligand_coords - predicted_center) * resolution, axis=1)
    return float(distances.min()) if distances.size else math.inf


def _component_mask(labeled_components, component_id):
    if labeled_components is None or component_id is None:
        return None
    return labeled_components == component_id


def _resolve_dcc_reference(label_center, ligand_center, mode):
    mode = (mode or "label_center").strip().lower()
    if mode in {"label", "label_center", "binding_site", "actual_binding_site"}:
        if label_center is not None:
            return label_center, "label_center"
        return ligand_center, "ligand_center"
    if mode in {"ligand", "ligand_center"}:
        if ligand_center is not None:
            return ligand_center, "ligand_center"
        return label_center, "label_center"
    if mode == "auto":
        if ligand_center is not None:
            return ligand_center, "ligand_center"
        return label_center, "label_center"
    raise ValueError(
        "dcc_reference must be one of label_center, ligand_center, or auto; "
        f"got {mode!r}"
    )


def _dvo(component_mask, label_mask):
    if component_mask is None:
        return 0.0
    label_bool = np.asarray(label_mask) > 0.5
    union = np.logical_or(component_mask, label_bool).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(component_mask, label_bool).sum()
    return float(intersection / union)


def _pli(component_mask, ligand_mask):
    """Proportion of ligand voxels included in the predicted binding site."""
    if component_mask is None or ligand_mask is None:
        return 0.0
    ligand_bool = np.asarray(ligand_mask) > 0.5
    ligand_volume = int(ligand_bool.sum())
    if ligand_volume == 0:
        return 0.0
    intersection = np.logical_and(component_mask, ligand_bool).sum()
    return float(intersection / ligand_volume)


def evaluate_pocket_metrics_for_sample(
    probabilities,
    label_mask,
    ligand_mask,
    protein_name,
    thresholds,
    resolution,
    max_distance_angstrom,
    dcc_cutoff_angstrom=4.0,
    dca_cutoff_angstrom=4.0,
    min_component_voxels=5,
    min_component_volume_angstrom3=None,
    postprocess_mode=RAW_POSTPROCESS,
    top_k_values=(1, 3),
    dcc_reference="label_center",
):
    postprocess_mode = normalize_postprocess_mode(postprocess_mode)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    label_mask = np.asarray(label_mask)
    ligand_mask = None if ligand_mask is None else np.asarray(ligand_mask)
    resolution = float(resolution)
    max_distance_angstrom = float(max_distance_angstrom)
    max_components = max(top_k_values) if top_k_values else 1

    label_center, _ = center_of_mask(label_mask)
    ligand_center, ligand_coords = center_of_mask(ligand_mask) if ligand_mask is not None else (None, None)
    dcc_reference_center, dcc_reference_label = _resolve_dcc_reference(
        label_center,
        ligand_center,
        dcc_reference,
    )

    rows = []
    for threshold in thresholds:
        components, component_count, predicted_positive_voxels, labeled = extract_components(
            probabilities,
            threshold,
            min_component_voxels=min_component_voxels,
            max_components=max_components,
            postprocess_mode=postprocess_mode,
            resolution=resolution,
            min_component_volume_angstrom3=min_component_volume_angstrom3,
        )
        top_component = components[0] if components else None
        top_component_mask = _component_mask(labeled, top_component["component_id"]) if top_component else None
        predicted = top_component is not None

        if predicted:
            predicted_center = top_component["center"]
            dcc = _distance_angstrom(predicted_center, dcc_reference_center, resolution)
            dcc_to_label = _distance_angstrom(predicted_center, label_center, resolution)
            dcc_to_ligand = _distance_angstrom(predicted_center, ligand_center, resolution)
            dca = _dca_angstrom(predicted_center, ligand_coords, resolution)
            dvo = _dvo(top_component_mask, label_mask)
            pli = _pli(top_component_mask, ligand_mask)
            top_component_voxels = top_component["voxel_count"]
        else:
            dcc = max_distance_angstrom
            dcc_to_label = max_distance_angstrom
            dcc_to_ligand = max_distance_angstrom
            dca = max_distance_angstrom
            dvo = 0.0
            pli = 0.0
            top_component_voxels = 0

        top_k_success = {}
        for top_k in top_k_values:
            success = False
            for component in components[:top_k]:
                component_dcc = _distance_angstrom(component["center"], dcc_reference_center, resolution)
                if component_dcc <= dcc_cutoff_angstrom:
                    success = True
                    break
            top_k_success[top_k] = success

        dcc_success = bool(dcc <= dcc_cutoff_angstrom)
        dca_success = bool(dca <= dca_cutoff_angstrom)

        paper_tp = int(predicted and dcc_success)
        paper_fp = int(predicted and not dcc_success)
        paper_fn = int(not predicted)

        strict_tp = int(predicted and dcc_success)
        strict_fp = int(predicted and not dcc_success)
        strict_fn = int(not predicted or (predicted and not dcc_success))

        rows.append(
            {
                "protein": protein_name,
                "threshold": float(threshold),
                "predicted": int(predicted),
                "component_count": int(component_count),
                "top_component_voxels": int(top_component_voxels),
                "predicted_positive_voxels": int(predicted_positive_voxels),
                "dcc_angstrom": float(dcc),
                "dcc_to_label_angstrom": float(dcc_to_label),
                "dcc_to_ligand_angstrom": float(dcc_to_ligand),
                "dca_angstrom": float(dca),
                "dvo": float(dvo),
                "pli": float(pli),
                "dcc_success_4a": int(dcc_success),
                "dca_success_4a": int(dca_success),
                "top1_dcc_success_4a": int(top_k_success.get(1, dcc_success)),
                "top3_dcc_success_4a": int(top_k_success.get(3, dcc_success)),
                "paper_tp": paper_tp,
                "paper_fp": paper_fp,
                "paper_fn": paper_fn,
                "strict_tp": strict_tp,
                "strict_fp": strict_fp,
                "strict_fn": strict_fn,
                "dcc_reference": dcc_reference_label,
            }
        )

    return rows


def evaluate_topk_metrics_for_sample(
    probabilities,
    label_mask,
    ligand_mask,
    protein_name,
    thresholds,
    resolution,
    max_distance_angstrom,
    dcc_cutoff_angstrom=4.0,
    dca_cutoff_angstrom=4.0,
    min_component_voxels=5,
    min_component_volume_angstrom3=None,
    postprocess_mode=RAW_POSTPROCESS,
    top_k_values=(1, 2, 3),
    reference_pocket_count=1,
    include_top_n_plus_2=True,
    dcc_reference="label_center",
):
    postprocess_mode = normalize_postprocess_mode(postprocess_mode)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    label_mask = np.asarray(label_mask)
    ligand_mask = None if ligand_mask is None else np.asarray(ligand_mask)
    resolution = float(resolution)
    max_distance_angstrom = float(max_distance_angstrom)
    reference_pocket_count = max(1, int(reference_pocket_count))

    top_k_specs = [(f"top{int(top_k)}", int(top_k)) for top_k in top_k_values]
    if include_top_n_plus_2:
        top_k_specs.append(("top_n_plus_2", reference_pocket_count + 2))

    max_components = max((top_k for _, top_k in top_k_specs), default=1)
    label_center, _ = center_of_mask(label_mask)
    ligand_center, ligand_coords = center_of_mask(ligand_mask) if ligand_mask is not None else (None, None)
    dcc_reference_center, dcc_reference_label = _resolve_dcc_reference(
        label_center,
        ligand_center,
        dcc_reference,
    )

    component_rows = []
    topk_rows = []

    for threshold in thresholds:
        components, component_count, predicted_positive_voxels, labeled = extract_components(
            probabilities,
            threshold,
            min_component_voxels=min_component_voxels,
            max_components=max_components,
            postprocess_mode=postprocess_mode,
            resolution=resolution,
            min_component_volume_angstrom3=min_component_volume_angstrom3,
        )

        ranked_components = []
        for rank, component in enumerate(components, start=1):
            component_mask = _component_mask(labeled, component["component_id"])
            dcc = _distance_angstrom(component["center"], dcc_reference_center, resolution)
            dcc_to_label = _distance_angstrom(component["center"], label_center, resolution)
            dcc_to_ligand = _distance_angstrom(component["center"], ligand_center, resolution)
            dca = _dca_angstrom(component["center"], ligand_coords, resolution)
            dvo = _dvo(component_mask, label_mask)
            pli = _pli(component_mask, ligand_mask)
            center = component["center"]
            component_metric = {
                "protein": protein_name,
                "threshold": float(threshold),
                "component_rank": int(rank),
                "component_id": int(component["component_id"]),
                "component_voxels": int(component["voxel_count"]),
                "score_sum": float(component["score_sum"]),
                "score_mean": float(component["score_mean"]),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "dcc_angstrom": float(dcc),
                "dcc_to_label_angstrom": float(dcc_to_label),
                "dcc_to_ligand_angstrom": float(dcc_to_ligand),
                "dca_angstrom": float(dca),
                "dvo": float(dvo),
                "pli": float(pli),
                "dcc_success_4a": int(dcc <= dcc_cutoff_angstrom),
                "dca_success_4a": int(dca <= dca_cutoff_angstrom),
                "dcc_reference": dcc_reference_label,
            }
            component_rows.append(component_metric)
            ranked_components.append(component_metric)

        for top_k_label, top_k in top_k_specs:
            candidates = ranked_components[:top_k]
            predicted = bool(candidates)

            if candidates:
                best_dcc_component = min(candidates, key=lambda row: row["dcc_angstrom"])
                best_dca_component = min(candidates, key=lambda row: row["dca_angstrom"])
                best_dvo_component = max(candidates, key=lambda row: row["dvo"])
                best_pli_component = max(candidates, key=lambda row: row["pli"])
                best_dcc = float(best_dcc_component["dcc_angstrom"])
                best_dca = float(best_dca_component["dca_angstrom"])
                best_dvo = float(best_dvo_component["dvo"])
                best_pli = float(best_pli_component["pli"])
                dvo_of_best_dcc = float(best_dcc_component["dvo"])
                dvo_of_best_dca = float(best_dca_component["dvo"])
                pli_of_best_dcc = float(best_dcc_component["pli"])
                pli_of_best_dca = float(best_dca_component["pli"])
                best_dcc_rank = int(best_dcc_component["component_rank"])
                best_dca_rank = int(best_dca_component["component_rank"])
                best_dvo_rank = int(best_dvo_component["component_rank"])
                best_pli_rank = int(best_pli_component["component_rank"])
                dcc_of_best_dvo = float(best_dvo_component["dcc_angstrom"])
                dca_of_best_dvo = float(best_dvo_component["dca_angstrom"])
                pli_of_best_dvo = float(best_dvo_component["pli"])
                dcc_of_best_pli = float(best_pli_component["dcc_angstrom"])
                dca_of_best_pli = float(best_pli_component["dca_angstrom"])
                dvo_of_best_pli = float(best_pli_component["dvo"])
            else:
                best_dcc = max_distance_angstrom
                best_dca = max_distance_angstrom
                best_dvo = 0.0
                best_pli = 0.0
                dvo_of_best_dcc = 0.0
                dvo_of_best_dca = 0.0
                pli_of_best_dcc = 0.0
                pli_of_best_dca = 0.0
                best_dcc_rank = 0
                best_dca_rank = 0
                best_dvo_rank = 0
                best_pli_rank = 0
                dcc_of_best_dvo = max_distance_angstrom
                dca_of_best_dvo = max_distance_angstrom
                pli_of_best_dvo = 0.0
                dcc_of_best_pli = max_distance_angstrom
                dca_of_best_pli = max_distance_angstrom
                dvo_of_best_pli = 0.0

            dcc_success = bool(best_dcc <= dcc_cutoff_angstrom)
            dca_success = bool(best_dca <= dca_cutoff_angstrom)
            best_dvo_dcc_success = bool(dcc_of_best_dvo <= dcc_cutoff_angstrom)
            best_dvo_dca_success = bool(dca_of_best_dvo <= dca_cutoff_angstrom)
            best_pli_dcc_success = bool(dcc_of_best_pli <= dcc_cutoff_angstrom)
            best_pli_dca_success = bool(dca_of_best_pli <= dca_cutoff_angstrom)
            topk_rows.append(
                {
                    "protein": protein_name,
                    "threshold": float(threshold),
                    "top_k_label": top_k_label,
                    "top_k": int(top_k),
                    "reference_pocket_count": int(reference_pocket_count),
                    "predicted": int(predicted),
                    "candidate_count": int(len(candidates)),
                    "component_count": int(component_count),
                    "predicted_positive_voxels": int(predicted_positive_voxels),
                    "best_dcc_angstrom": float(best_dcc),
                    "best_dcc_component_rank": int(best_dcc_rank),
                    "dvo_of_best_dcc_component": float(dvo_of_best_dcc),
                    "pli_of_best_dcc_component": float(pli_of_best_dcc),
                    "best_dca_angstrom": float(best_dca),
                    "best_dca_component_rank": int(best_dca_rank),
                    "dvo_of_best_dca_component": float(dvo_of_best_dca),
                    "pli_of_best_dca_component": float(pli_of_best_dca),
                    "best_dvo": float(best_dvo),
                    "best_dvo_component_rank": int(best_dvo_rank),
                    "dcc_of_best_dvo_component": float(dcc_of_best_dvo),
                    "dca_of_best_dvo_component": float(dca_of_best_dvo),
                    "pli_of_best_dvo_component": float(pli_of_best_dvo),
                    "best_pli": float(best_pli),
                    "best_pli_component_rank": int(best_pli_rank),
                    "dcc_of_best_pli_component": float(dcc_of_best_pli),
                    "dca_of_best_pli_component": float(dca_of_best_pli),
                    "dvo_of_best_pli_component": float(dvo_of_best_pli),
                    "best_dvo_dcc_success_4a": int(best_dvo_dcc_success),
                    "best_dvo_dca_success_4a": int(best_dvo_dca_success),
                    "best_pli_dcc_success_4a": int(best_pli_dcc_success),
                    "best_pli_dca_success_4a": int(best_pli_dca_success),
                    "dcc_success_4a": int(dcc_success),
                    "dca_success_4a": int(dca_success),
                    "paper_tp": int(predicted and dcc_success),
                    "paper_fp": int(predicted and not dcc_success),
                    "paper_fn": int(not predicted),
                    "dcc_reference": dcc_reference_label,
                }
            )

    return component_rows, topk_rows


def summarize_topk_pocket_metrics(rows):
    rows_by_key = defaultdict(list)
    for row in rows:
        key = (
            row.get("epoch", ""),
            row.get("run", ""),
            row.get("checkpoint", ""),
            row.get("postprocess_mode", ""),
            float(row["threshold"]),
            row["top_k_label"],
            int(row["top_k"]),
        )
        rows_by_key[key].append(row)

    summaries = []
    for (
        epoch,
        run,
        checkpoint,
        postprocess_mode,
        threshold,
        top_k_label,
        top_k,
    ), grouped_rows in sorted(rows_by_key.items()):
        sample_count = len(grouped_rows)
        paper_tp = sum(int(row["paper_tp"]) for row in grouped_rows)
        paper_fp = sum(int(row["paper_fp"]) for row in grouped_rows)
        paper_fn = sum(int(row["paper_fn"]) for row in grouped_rows)
        precision, recall, pocket_f1 = calculate_f1_from_counts(paper_tp, paper_fp, paper_fn)
        dcc_success_rows = [row for row in grouped_rows if int(row["dcc_success_4a"])]

        def mean(key):
            values = np.asarray([float(row[key]) for row in grouped_rows], dtype=np.float64)
            return float(values.mean()) if values.size else 0.0

        dvo_success_values = np.asarray(
            [float(row["dvo_of_best_dcc_component"]) for row in dcc_success_rows],
            dtype=np.float64,
        )
        pli_success_values = np.asarray(
            [float(row["pli_of_best_dcc_component"]) for row in dcc_success_rows],
            dtype=np.float64,
        )
        best_pli_success_rows = [row for row in grouped_rows if int(row["best_pli_dcc_success_4a"])]
        best_pli_success_values = np.asarray(
            [float(row["best_pli"]) for row in best_pli_success_rows],
            dtype=np.float64,
        )
        first_row = grouped_rows[0]
        summaries.append(
            {
                "epoch": epoch,
                "run": run,
                "model": first_row.get("model", ""),
                "feature_family": first_row.get("feature_family", ""),
                "apbs_variant": first_row.get("apbs_variant", ""),
                "checkpoint": checkpoint,
                "postprocess_mode": postprocess_mode,
                "threshold": threshold,
                "top_k_label": top_k_label,
                "top_k": int(top_k),
                "pocket_f1": pocket_f1,
                "pocket_precision": precision,
                "pocket_recall": recall,
                "paper_tp": int(paper_tp),
                "paper_fp": int(paper_fp),
                "paper_fn": int(paper_fn),
                "dcc_success_rate_4a": sum(int(row["dcc_success_4a"]) for row in grouped_rows)
                / sample_count,
                "dca_success_rate_4a": sum(int(row["dca_success_4a"]) for row in grouped_rows)
                / sample_count,
                "mean_best_dcc_angstrom": mean("best_dcc_angstrom"),
                "mean_best_dca_angstrom": mean("best_dca_angstrom"),
                "mean_best_dvo_all": mean("best_dvo"),
                "mean_best_pli_all": mean("best_pli"),
                "mean_dcc_of_best_dvo_angstrom": mean("dcc_of_best_dvo_component"),
                "mean_dca_of_best_dvo_angstrom": mean("dca_of_best_dvo_component"),
                "mean_dcc_of_best_pli_angstrom": mean("dcc_of_best_pli_component"),
                "mean_dca_of_best_pli_angstrom": mean("dca_of_best_pli_component"),
                "best_dvo_dcc_success_rate_4a": sum(
                    int(row["best_dvo_dcc_success_4a"]) for row in grouped_rows
                )
                / sample_count,
                "best_dvo_dca_success_rate_4a": sum(
                    int(row["best_dvo_dca_success_4a"]) for row in grouped_rows
                )
                / sample_count,
                "best_pli_dcc_success_rate_4a": sum(
                    int(row["best_pli_dcc_success_4a"]) for row in grouped_rows
                )
                / sample_count,
                "best_pli_dca_success_rate_4a": sum(
                    int(row["best_pli_dca_success_4a"]) for row in grouped_rows
                )
                / sample_count,
                "mean_dvo_of_best_dcc_all": mean("dvo_of_best_dcc_component"),
                "mean_dvo_of_best_dcc_success": (
                    float(dvo_success_values.mean()) if dvo_success_values.size else 0.0
                ),
                "mean_pli_of_best_dcc_all": mean("pli_of_best_dcc_component"),
                "mean_pli_of_best_dcc_success": (
                    float(pli_success_values.mean()) if pli_success_values.size else 0.0
                ),
                "mean_best_pli_dcc_success": (
                    float(best_pli_success_values.mean()) if best_pli_success_values.size else 0.0
                ),
                "mean_predicted_positive_voxels": mean("predicted_positive_voxels"),
                "mean_candidate_count": mean("candidate_count"),
                "no_prediction_count": int(sum(1 for row in grouped_rows if not int(row["predicted"]))),
                "sample_count": int(sample_count),
            }
        )
    return summaries


def calculate_selection_score(
    summary_row,
    selection_metric="paper_f1",
    dvo_weight=5.0,
    voxel_f1=0.0,
    voxel_f1_weight=1.0,
    dca_weight=0.25,
    no_dcc_score_scale=0.05,
    max_mean_predicted_positive_voxels=None,
):
    base_score = float(summary_row["paper_f1"])
    if selection_metric == "paper_f1":
        return base_score

    dvo_for_selection = float(summary_row.get("mean_dvo_dcc_success", summary_row.get("mean_dvo_all", 0.0)))
    score = base_score * (1.0 + float(dvo_weight) * dvo_for_selection)
    if selection_metric == "dcc_dvo":
        return score
    if selection_metric == "dcc_dvo_volume":
        if max_mean_predicted_positive_voxels is None:
            return score
        mean_predicted = max(float(summary_row["mean_predicted_positive_voxels"]), 1.0)
        volume_penalty = min(1.0, float(max_mean_predicted_positive_voxels) / mean_predicted)
        return score * volume_penalty
    if selection_metric == "dcc_voxel_dca_dvo_volume":
        dca_reward = float(dca_weight) * float(summary_row["dca_success_rate_4a"])
        dvo_reward = float(dvo_weight) * float(summary_row["mean_dvo_all"])
        voxel_reward = float(voxel_f1_weight) * float(voxel_f1)
        dcc_gate = 1.0 if base_score > 0.0 else float(no_dcc_score_scale)
        score = base_score + dcc_gate * (voxel_reward + dca_reward + dvo_reward)
        if max_mean_predicted_positive_voxels is None:
            return score
        mean_predicted = max(float(summary_row["mean_predicted_positive_voxels"]), 1.0)
        volume_penalty = min(1.0, float(max_mean_predicted_positive_voxels) / mean_predicted)
        return score * volume_penalty
    raise ValueError(f"Unsupported paper metric selection_metric: {selection_metric}")


def summarize_pocket_metrics(
    rows,
    thresholds,
    selection_metric="paper_f1",
    selection_dvo_weight=5.0,
    selection_voxel_f1_weight=1.0,
    selection_dca_weight=0.25,
    selection_no_dcc_score_scale=0.05,
    selection_max_mean_predicted_positive_voxels=None,
    voxel_summary_by_threshold=None,
):
    voxel_summary_by_threshold = voxel_summary_by_threshold or {}
    rows_by_threshold = defaultdict(list)
    for row in rows:
        rows_by_threshold[float(row["threshold"])].append(row)

    summaries = []
    for threshold in thresholds:
        threshold = float(threshold)
        threshold_rows = rows_by_threshold.get(threshold, [])
        sample_count = len(threshold_rows)
        if sample_count == 0:
            continue

        paper_tp = sum(row["paper_tp"] for row in threshold_rows)
        paper_fp = sum(row["paper_fp"] for row in threshold_rows)
        paper_fn = sum(row["paper_fn"] for row in threshold_rows)
        paper_precision, paper_recall, paper_f1 = calculate_f1_from_counts(paper_tp, paper_fp, paper_fn)

        strict_tp = sum(row["strict_tp"] for row in threshold_rows)
        strict_fp = sum(row["strict_fp"] for row in threshold_rows)
        strict_fn = sum(row["strict_fn"] for row in threshold_rows)
        strict_precision, strict_recall, strict_f1 = calculate_f1_from_counts(strict_tp, strict_fp, strict_fn)

        dcc_values = np.asarray([row["dcc_angstrom"] for row in threshold_rows], dtype=np.float64)
        dcc_to_label_values = np.asarray(
            [row["dcc_to_label_angstrom"] for row in threshold_rows], dtype=np.float64
        )
        dca_values = np.asarray([row["dca_angstrom"] for row in threshold_rows], dtype=np.float64)
        dvo_values = np.asarray([row["dvo"] for row in threshold_rows], dtype=np.float64)
        pli_values = np.asarray([row["pli"] for row in threshold_rows], dtype=np.float64)
        dcc_success_rows = [row for row in threshold_rows if row["dcc_success_4a"]]
        dvo_success_values = np.asarray([row["dvo"] for row in dcc_success_rows], dtype=np.float64)
        pli_success_values = np.asarray([row["pli"] for row in dcc_success_rows], dtype=np.float64)
        predicted_voxels = np.asarray(
            [row["predicted_positive_voxels"] for row in threshold_rows], dtype=np.float64
        )
        top_component_voxels = np.asarray(
            [row["top_component_voxels"] for row in threshold_rows], dtype=np.float64
        )

        summary_row = {
            "threshold": threshold,
            "paper_f1": paper_f1,
            "paper_precision": paper_precision,
            "paper_recall": paper_recall,
            "paper_tp": int(paper_tp),
            "paper_fp": int(paper_fp),
            "paper_fn": int(paper_fn),
            "strict_f1": strict_f1,
            "strict_precision": strict_precision,
            "strict_recall": strict_recall,
            "strict_tp": int(strict_tp),
            "strict_fp": int(strict_fp),
            "strict_fn": int(strict_fn),
            "dcc_success_rate_4a": sum(row["dcc_success_4a"] for row in threshold_rows) / sample_count,
            "dca_success_rate_4a": sum(row["dca_success_4a"] for row in threshold_rows) / sample_count,
            "top1_dcc_success_rate_4a": sum(row["top1_dcc_success_4a"] for row in threshold_rows)
            / sample_count,
            "top3_dcc_success_rate_4a": sum(row["top3_dcc_success_4a"] for row in threshold_rows)
            / sample_count,
            "mean_dcc_angstrom": float(dcc_values.mean()),
            "median_dcc_angstrom": float(np.median(dcc_values)),
            "mean_dcc_to_label_angstrom": float(dcc_to_label_values.mean()),
            "mean_dca_angstrom": float(dca_values.mean()),
            "mean_dvo_all": float(dvo_values.mean()),
            "mean_dvo_dcc_success": float(dvo_success_values.mean()) if dvo_success_values.size else 0.0,
            "mean_pli_all": float(pli_values.mean()),
            "mean_pli_dcc_success": float(pli_success_values.mean()) if pli_success_values.size else 0.0,
            "dcc_success_count": int(len(dcc_success_rows)),
            "mean_predicted_positive_voxels": float(predicted_voxels.mean()),
            "mean_top_component_voxels": float(top_component_voxels.mean()),
            "no_prediction_count": int(sum(1 for row in threshold_rows if not row["predicted"])),
            "sample_count": int(sample_count),
            "selection_metric": selection_metric,
        }
        summary_row["selection_score"] = calculate_selection_score(
            summary_row,
            selection_metric=selection_metric,
            dvo_weight=selection_dvo_weight,
            voxel_f1=voxel_summary_by_threshold.get(threshold, {}).get("f1", 0.0),
            voxel_f1_weight=selection_voxel_f1_weight,
            dca_weight=selection_dca_weight,
            no_dcc_score_scale=selection_no_dcc_score_scale,
            max_mean_predicted_positive_voxels=selection_max_mean_predicted_positive_voxels,
        )
        summaries.append(summary_row)

    return summaries


def select_best_paper_summary(summary_rows):
    if not summary_rows:
        return None
    return max(
        summary_rows,
        key=lambda row: (
            row["selection_score"],
            row["paper_f1"],
            row["dcc_success_rate_4a"],
            row["mean_dvo_dcc_success"],
            -row["mean_dcc_angstrom"],
        ),
    )
