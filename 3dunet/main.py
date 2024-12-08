import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Gerekli sınıflar ve fonksiyonlar
from dataset import ProteinLigandDatasetWithH5
from losses import BCEDiceLoss
from model import UNet3D
from transforms import RandomFlip, RandomRotate3D, Standardize, CustomCompose


# Argümanları al
def parse_args():
    parser = argparse.ArgumentParser(description="3D U-Net Training Script")
    parser.add_argument(
        "--config", type=str, default="config/config.yml", help="Path to the config file"
    )
    return parser.parse_args()


# Config dosyasını yükle
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Ana kod
if __name__ == "__main__":
    # Argümanları al ve config'i yükle
    args = parse_args()
    config = load_config(args.config)

    # Config'ten değerleri yükle
    data_directory = config["data_directory"]
    cache_directory = config["cache_directory"]
    train_proteins = config["datasets"]["train"]
    validation_proteins = config["datasets"]["validation"]
    test_proteins = config["datasets"]["test"]
    h5_directory = config["h5_directory"]

    # # Dataset'leri oluştur
    # train_dataset = ProteinLigandDataset(
    #     root_dir=data_directory, cache_dir=cache_directory,
    #     protein_names=train_proteins,
    #     transform=CustomCompose([RandomFlip(), RandomRotate3D(), Standardize()])
    # )
    # validation_dataset = ProteinLigandDataset(
    #     root_dir=data_directory, cache_dir=cache_directory,
    #     protein_names=validation_proteins,
    #     transform=CustomCompose([Standardize()])
    # )
    # test_dataset = ProteinLigandDataset(
    #     root_dir=data_directory, cache_dir=cache_directory,
    #     protein_names=test_proteins,
    #     transform=CustomCompose([Standardize()])
    # )

    # Dataset'leri oluştur
    train_dataset = ProteinLigandDatasetWithH5(
        h5_dir=h5_directory,
        protein_names=train_proteins,
        transform=CustomCompose([RandomFlip(), RandomRotate3D(), Standardize()])
    )
    validation_dataset = ProteinLigandDatasetWithH5(
        h5_dir=h5_directory,
        protein_names=validation_proteins,
        transform=CustomCompose([Standardize()])
    )
    test_dataset = ProteinLigandDatasetWithH5(
        h5_dir=h5_directory,
        protein_names=test_proteins,
        transform=CustomCompose([Standardize()])
    )

    # CPU çekirdek sayısını belirle
    cpu_count = os.cpu_count()

    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=10)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=10)

    # Model, kayıp fonksiyonu, optimizer ve scheduler tanımlamaları
    device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D()

    # Çoklu GPU kontrolü
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Model özetini göster
   # summary(model, input_size=(4, 2, 161, 161, 161))  # Batch size, channels, depth, height, width

    pos_weight = torch.tensor([10.0]).to(device)
    criterion = BCEDiceLoss(alpha=0.5, smooth=1.0, pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

    # TensorBoard'u başlat
    writer = SummaryWriter("runs/3d_unet_experiment")

    # Eğitim döngüsü
    num_epochs = config.get("num_epochs", 100)
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_targets, all_predictions = [], []
        total_batches = len(train_loader)
        for batch_idx, (protein, ligand) in enumerate(train_loader, start=1):
            protein, ligand = protein.to(device), ligand.to(device)
            optimizer.zero_grad()
            output = model(protein)
            output = output.squeeze(1)
            loss = criterion(output, ligand)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}")

            # F1 skorunu hesapla
            output_probs = torch.sigmoid(output).detach().cpu().numpy()
            output_preds = (output_probs > 0.4).astype(np.uint8)
            targets = ligand.cpu().numpy().astype(np.uint8)
            all_targets.extend(targets.flatten())
            all_predictions.extend(output_preds.flatten())

        train_f1 = f1_score(all_targets, all_predictions, average='binary')
        print(f"Training Loss: {total_loss / len(train_loader)}, Training F1 Score: {train_f1:.4f}")

        # Validation modu
        model.eval()
        validation_loss = 0
        val_targets, val_predictions = [], []

        print(f"Epoch {epoch + 1}/{num_epochs} - Validation...")
        with torch.no_grad():
            for batch_idx, (protein, ligand) in enumerate(validation_loader, start=1):
                print(f"Validation Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(validation_loader)}")
                protein, ligand = protein.to(device), ligand.to(device)
                output = model(protein).squeeze(1)
                loss = criterion(output, ligand)
                validation_loss += loss.item()

                # F1 skoru için verileri topla
                output_probs = torch.sigmoid(output).detach().cpu().numpy()
                output_preds = (output_probs > 0.4).astype(np.uint8)
                targets = ligand.cpu().numpy().astype(np.uint8)
                val_targets.extend(targets.flatten())
                val_predictions.extend(output_preds.flatten())

        val_f1 = f1_score(val_targets, val_predictions, average='binary')
        print(f"Validation Loss: {validation_loss / len(validation_loader)}, Validation F1 Score: {val_f1:.4f}")

        # Öğrenme oranını azaltıcıyı güncelle
        scheduler.step(validation_loss / len(validation_loader))

        # TensorBoard'a değerleri yaz
        writer.add_scalar("Loss/Train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Validation", validation_loss / len(validation_loader), epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)

        # scheduler.step(total_loss / len(train_loader))

    # TensorBoard'u kapat
    writer.close()
    print("Eğitim tamamlandı!")
