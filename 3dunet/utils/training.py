import torch
import torch.optim as optim
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