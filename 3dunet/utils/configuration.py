import argparse

import yaml


def parse_args():
    """
    Argümanları okuyarak döner.
    """
    parser = argparse.ArgumentParser(description="3D U-Net Training Script")
    parser.add_argument(
        "--config", type=str, default="config/local/config.yml", help="Path to the config file"
    )
    parser.add_argument("--model", type=str, default="UNet3D4L", help="Model class name")
    parser.add_argument("--base_features", type=int, default=64, help="Base features")
    parser.add_argument("--num_workers", type=int, default=16, help="Worker count")
    return parser.parse_args()


def load_config(config_path):
    """
    Config dosyasını yükler.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def create_output_dirs(base_dir, config_name):
    """
    Log, ağırlık ve TensorBoard dosyalarını kaydetmek için dizinleri oluşturur.
    """
    log_dir = os.path.join(base_dir, config_name, "log")
    weights_dir = os.path.join(base_dir, config_name, "weights")
    tensorboard_dir = os.path.join(base_dir, config_name, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    return log_dir, weights_dir, tensorboard_dir


import logging
import os


def setup_logger(log_dir, log_filename="training.log", level=logging.INFO):
    """
    Konsola ve dosyaya log yazan bir logger ayarlama fonksiyonu.

    Args:
        log_dir (str): Log dosyasının yazılacağı dizin.
        log_filename (str): Log dosyasının adı.
        level (int): Loglama seviyesi (örn: logging.INFO).

    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    # Log dizinini oluştur
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Kök logger'ı temizle
    logging.getLogger().handlers.clear()

    # Logger'ı oluştur
    logger = logging.getLogger("training_logger")
    logger.setLevel(level)

    if not logger.hasHandlers():
        # Konsol için handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Dosya için handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
