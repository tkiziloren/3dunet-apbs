import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchmetrics.classification import F1Score, Precision, Recall, ConfusionMatrix

from .losses import BCEDiceLoss


def calculate_pos_weight_from_loader(train_loader, max_batches=50):
    """
    Calculate optimal pos_weight dynamically from training data
    
    Args:
        train_loader: Training data loader
        max_batches: Maximum number of batches to sample (for speed)
    
    Returns:
        float: Calculated pos_weight (capped at 100)
    """
    total_pos = 0
    total_voxels = 0
    
    print("Calculating pos_weight from training data...")
    for i, (protein, label) in enumerate(train_loader):
        if i >= max_batches:
            break
        total_pos += label.sum().item()
        total_voxels += label.numel()
    
    if total_voxels == 0:
        return 10.0  # Default fallback
    
    ratio = total_pos / total_voxels
    pos_weight = (1 - ratio) / ratio if ratio > 0 else 10.0
    pos_weight = min(pos_weight, 100.0)  # Cap at 100
    
    print(f"  Positive voxel ratio: {ratio:.6f}")
    print(f"  Calculated pos_weight: {pos_weight:.1f}")
    
    return pos_weight


def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    """
    Verilen parametrelere göre optimizer oluşturur.
    """
    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(scheduler_config, optimizer):
    """
    Scheduler oluşturur.
    """
    scheduler_type = scheduler_config.get("type")
    if scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.2),
            patience=scheduler_config.get("patience", 10)
        )
    elif scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_loss_function(loss_config, device):
    """
    Loss fonksiyonu oluşturur.
    """
    loss_type = loss_config.get("type")
    if loss_type == "BCEDiceLoss":
        return BCEDiceLoss(
            alpha=loss_config.get("alpha", 0.5),
            smooth=loss_config.get("smooth", 1.0),
            pos_weight=torch.tensor([loss_config.get("pos_weight", 1.0)]).to(device)
        )
    elif loss_type == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif loss_type == "MSELoss":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Metal Performance Shaders (Apple GPU)
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def initialize_metrics(threshold=0.5, device=None):
    """
    F1, Precision, Recall ve Confusion Matrix metriklerini bir kez başlatır.

    Args:
        threshold (float): Sigmoid sonrası tahmin eşiği.
        device (str): Kullanılacak cihaz (default: otomatik seç).

    Returns:
        tuple: F1, Precision, Recall, ConfusionMatrix objeleri.
    """
    if device is None:
        device = get_device()

    f1_metric = F1Score(task='binary', threshold=threshold).to(device)
    precision_metric = Precision(task='binary', threshold=threshold).to(device)
    recall_metric = Recall(task='binary', threshold=threshold).to(device)
    confusion_matrix_metric = ConfusionMatrix(task="binary", threshold=threshold).to(device)
    return f1_metric, precision_metric, recall_metric, confusion_matrix_metric


def calculate_metrics_sklearn(targets, preds):
    f1 = f1_score(targets, preds, average='binary')
    precision = precision_score(targets, preds, average='binary')
    recall = recall_score(targets, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return f1, precision, recall, (tp, fp, tn, fn)


def calculate_metrics_torchmetrics(targets, preds, metrics):
    f1_metric, precision_metric, recall_metric, confusion_matrix_metric = metrics

    # Metrikleri hesapla
    f1 = f1_metric(preds, targets).item()
    precision = precision_metric(preds, targets).item()
    recall = recall_metric(preds, targets).item()

    # Confusion matrix değerlerini al
    cm = confusion_matrix_metric(preds, targets).cpu().numpy()
    tn, fp, fn, tp = cm.ravel()

    return f1, precision, recall, (tp, fp, tn, fn)


def calculate_f1_sklearn(targets, preds):
    return f1_score(targets, preds, average='binary')


def calculate_f1_torchmetrics(targets, preds, metrics):
    f1_metric, precision_metric, recall_metric, confusion_matrix_metric = metrics
    return f1_metric(preds, targets).item()


def calculate_metrics(pocket_label, output, metrics, threshold=0.5):
    if get_device() == "cuda":
        predictions = torch.sigmoid(output).detach() > threshold
        return calculate_metrics_torchmetrics(pocket_label, predictions, metrics)
    else:
        probs = torch.sigmoid(output).detach().cpu().numpy()
        predictions_flattened = (probs > threshold).astype(np.uint8).flatten()
        targets_flattened = pocket_label.cpu().numpy().flatten()
        return calculate_metrics_sklearn(targets_flattened, predictions_flattened)


def calculate_pocket_dcc(prediction, label, threshold=4.0):
    """
    Calculate pocket-level Distance Criterion Center (DCC) - LOCALIZATION metric.
    
    DCC_Localization measures the Euclidean distance between:
    - Center of mass of predicted binding site
    - Center of mass of true ligand/binding site
    
    Args:
        prediction: Model output (sigmoid applied, thresholded) - binary mask
        label: Ground truth binding site - binary mask
        threshold: Distance threshold in Angstroms (default 4.0Å)
    
    Returns:
        float: 1.0 if distance < threshold (success), 0.0 otherwise
        float: actual distance in voxels (for logging)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    
    # Binary predictions (threshold at 0.5)
    pred_binary = (prediction > 0.5).astype(bool)
    label_binary = (label > 0.5).astype(bool)
    
    # Calculate center of mass for predicted pocket
    pred_coords = np.argwhere(pred_binary)
    if len(pred_coords) == 0:
        # No prediction - maximum distance
        return 0.0, float('inf')
    pred_center = pred_coords.mean(axis=0)
    
    # Calculate center of mass for true binding site
    label_coords = np.argwhere(label_binary)
    if len(label_coords) == 0:
        # No true binding site (shouldn't happen)
        return 0.0, float('inf')
    label_center = label_coords.mean(axis=0)
    
    # Euclidean distance between centers (in voxels)
    distance = np.linalg.norm(pred_center - label_center)
    
    # Success if distance < threshold
    # Note: threshold is in Angstroms, but distance is in voxels
    # Assuming 1 voxel ≈ 1 Angstrom (adjust if needed)
    success = 1.0 if distance < threshold else 0.0
    
    return success, distance


def calculate_pocket_dcc_coverage(prediction, label):
    """
    Calculate DCC Coverage metric: TP / (TP + FN) at voxel level.
    
    DCC_Coverage measures how much of the true binding site is covered 
    by the prediction at the voxel level.
    
    Args:
        prediction: Model output (sigmoid applied, thresholded) - binary mask
        label: Ground truth binding site - binary mask
    
    Returns:
        float: Coverage ratio [0.0, 1.0] = TP / (TP + FN)
               1.0 means all true binding sites are correctly predicted
               0.0 means no true binding sites are predicted
    """
    # Convert to numpy if tensor
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    
    # Binary predictions (threshold at 0.5)
    pred_binary = (prediction > 0.5).astype(bool)
    label_binary = (label > 0.5).astype(bool)
    
    # If no label binding sites, coverage is undefined (return 0)
    if label_binary.sum() == 0:
        return 0.0
    
    # Calculate TP: overlap between prediction and label
    true_positives = np.logical_and(pred_binary, label_binary).sum()
    
    # Calculate FN: label sites not predicted
    false_negatives = np.logical_and(np.logical_not(pred_binary), label_binary).sum()
    
    # Coverage = TP / (TP + FN) = correctly predicted binding sites / all binding sites
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    
    coverage = true_positives / denominator
    return coverage
