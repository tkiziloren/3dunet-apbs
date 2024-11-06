import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ProteinLigandDataset ve Transform'ları import et
from dataset import ProteinLigandDataset
from losses import BCEDiceLoss
from model import UNet3D  # Modeli import et
from transforms import RandomFlip, RandomRotate3D, Standardize, CustomCompose

# PyYAML ile YAML dosyasını oku
with open("config/config.yml", "r") as file:
    config = yaml.safe_load(file)

data_directory = config["data_directory"]
cache_directory = config["cache_directory"]
train_proteins = config["datasets"]["train"]
validation_proteins = config["datasets"]["validation"]
test_proteins = config["datasets"]["test"]

# Veri setlerini oluştur
train_dataset = ProteinLigandDataset(
    root_dir=data_directory, cache_dir=cache_directory,
    protein_names=train_proteins,
    transform=CustomCompose([RandomFlip(), RandomRotate3D(), Standardize()])
)
validation_dataset = ProteinLigandDataset(
    root_dir=data_directory, cache_dir=cache_directory,
    protein_names=validation_proteins,
    transform=CustomCompose([Standardize()])
)
test_dataset = ProteinLigandDataset(
    root_dir=data_directory, cache_dir=cache_directory,
    protein_names=test_proteins,
    transform=CustomCompose([Standardize()])
)

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# Model, kayıp fonksiyonu, optimizer ve scheduler tanımlamaları
device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D().to(device)
criterion = BCEDiceLoss(1., 1.)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

# TensorBoard'u başlat
writer = SummaryWriter("runs/3d_unet_experiment")

# Eğitim döngüsü
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_targets, all_predictions = [], []

    for protein, ligand in train_loader:
        protein, ligand = protein.to(device), ligand.to(device)
        optimizer.zero_grad()
        output = model(protein)
        output = output.squeeze(1)
        loss = criterion(output, ligand)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # F1 skorunu hesapla
        output_probs = torch.sigmoid(output).detach().cpu().numpy()
        output_preds = (output_probs > 0.5).astype(np.uint8)
        targets = ligand.cpu().numpy().astype(np.uint8)
        all_targets.extend(targets.flatten())
        all_predictions.extend(output_preds.flatten())

        res = np.argmax(targets)
        # Benzersiz değerleri kontrol et
        unique_targets = np.unique(all_targets)
        unique_predictions = np.unique(all_predictions)

        print("Unique values in all_targets:", unique_targets)
        print("Unique values in all_predictions:", unique_predictions)

    f1 = f1_score(all_targets, all_predictions, average='binary')
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, F1 Score: {f1}")

    scheduler.step(total_loss / len(train_loader))

# TensorBoard'u kapat
writer.close()

print("Eğitim tamamlandı!")
