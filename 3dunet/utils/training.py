import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchmetrics.classification import F1Score, Precision, Recall, ConfusionMatrix

from .losses import BCEDiceLoss


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
        return "cpu"   # Metal Performance Shaders (Apple GPU)
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
