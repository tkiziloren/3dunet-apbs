import argparse
import logging
import os

import yaml


def parse_args():
    """
    Argümanları okuyarak döner.
    """
    parser = argparse.ArgumentParser(description="3D U-Net Training Script")
    parser.add_argument(
        "--config", type=str, default="config/local/config.yml", help="Path to the config file"
    )
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


def setup_logger(log_dir):
    """
    Logger yapılandırması. Hem dosyaya hem konsola log yazacak şekilde ayarlanır.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Log dosyasına yazma
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Konsola yazma
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Handlers ekle
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
