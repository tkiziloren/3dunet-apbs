import os
import argparse
import glob
import gzip
import shutil
import requests
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 0. Helpers for Filtering
# -----------------------------
def is_single_fragment(mol2_path):
    """Return True if mol2 file has exactly one disconnected fragment."""
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
    if mol is None:
        return False
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return len(frags) == 1

# -----------------------------
# 1. Data Download & Extraction
# -----------------------------
def download_and_extract(url: str, dest_folder: str):
    os.makedirs(dest_folder, exist_ok=True)
    gz_path = os.path.join(dest_folder, os.path.basename(url))
    if not os.path.exists(gz_path):
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("Archive already downloaded.")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(os.path.join(dest_folder, 'scPDB.mol2'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Extraction complete.")

# -----------------------------
# 2. Voxelizasyon & Caching
# -----------------------------

def mol2_to_voxel(mol2_path, atom_types, grid_size, res, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    types  = [atom.GetSymbol() for atom in mol.GetAtoms()]
    grid = np.zeros((len(atom_types), grid_size, grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    for (x,y,z), t in zip(coords, types):
        if t not in atom_types:
            continue
        idx = atom_types.index(t)
        i = int(round(x/res)) + center
        j = int(round(y/res)) + center
        k = int(round(z/res)) + center
        if 0 <= i < grid_size and 0 <= j < grid_size and 0 <= k < grid_size:
            grid[idx, i, j, k] = 1.0
    if cache_path:
        np.save(cache_path, grid)
    return grid

class ScPDBDataset(Dataset):
    def __init__(self, data_dir, atom_types, grid_size, res, indices, cache_dir=None):
        self.site_files = sorted(glob.glob(os.path.join(data_dir, 'site_*.mol2')))
        self.indices = indices
        self.atom_types = atom_types
        self.grid_size = grid_size
        self.res = res
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        idx = self.indices[ix]
        site_path = self.site_files[idx]
        ligand_path = site_path.replace('site_', 'ligand_')

        cache_x = cache_y = None
        if self.cache_dir:
            base = os.path.splitext(os.path.basename(site_path))[0]
            cache_x = os.path.join(self.cache_dir, f"{base}_x.npy")
            cache_y = os.path.join(self.cache_dir, f"{base}_y.npy")

        x = mol2_to_voxel(site_path, self.atom_types, self.grid_size, self.res, cache_path=cache_x)
        y_vox = mol2_to_voxel(ligand_path, self.atom_types, self.grid_size, self.res, cache_path=cache_y)
        y = (y_vox.sum(axis=0, keepdims=True) > 0).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------------
# 3. Model
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.dec1 = ConvBlock(64, 32)
        self.final = nn.Conv3d(32, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# -----------------------------
# 4. Metrics & Training/Eval
# -----------------------------
def dice_coeff(pred, target, eps=1e-6):
    pred_f = (torch.sigmoid(pred) > 0.5).float()
    inter = torch.sum(pred_f * target)
    union = torch.sum(pred_f) + torch.sum(target)
    return (2 * inter + eps) / (union + eps)

def iou_score(pred, target, eps=1e-6):
    pred_f = (torch.sigmoid(pred) > 0.5).float()
    inter = torch.sum(pred_f * target)
    union = torch.sum(pred_f) + torch.sum(target) - inter
    return (inter + eps) / (union + eps)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            total_dice += dice_coeff(pred, y).item()
            total_iou += iou_score(pred, y).item()
    return total_loss/len(loader), total_dice/len(loader), total_iou/len(loader)

# -----------------------------
# 5. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grid_size', type=int, default=64)
    parser.add_argument('--res', type=float, default=1.0)
    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # download_and_extract URL if needed

    # List all site files
    all_sites = sorted(glob.glob(os.path.join(args.data_dir, 'site_*.mol2')))
    # Filter to single-protein, single-ligand pairs
    valid_indices = []
    for i, site in enumerate(all_sites):
        ligand = site.replace('site_', 'ligand_')
        if is_single_fragment(site) and is_single_fragment(ligand):
            valid_indices.append(i)

    n = len(valid_indices)
    train_n = int(n * 0.8)
    val_n   = int(n * 0.1)
    train_idx = valid_indices[:train_n]
    val_idx   = valid_indices[train_n:train_n+val_n]
    test_idx  = valid_indices[train_n+val_n:]

    atom_types = ['C','N','O','S','P','H','Cl','Br','F','Fe']

    train_ds = ScPDBDataset(args.data_dir, atom_types, args.grid_size, args.res, train_idx, args.cache_dir)
    val_ds   = ScPDBDataset(args.data_dir, atom_types, args.grid_size, args.res, val_idx,   args.cache_dir)
    test_ds  = ScPDBDataset(args.data_dir, atom_types, args.grid_size, args.res, test_idx,  args.cache_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_ch=len(atom_types), out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}",
              f"Train Loss: {train_loss:.4f}",
              f"Val Loss: {val_loss:.4f}",
              f"Val Dice: {val_dice:.4f}",
              f"Val IoU: {val_iou:.4f}")
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pth')))
    test_loss, test_dice, test_iou = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}", f"Test Dice: {test_dice:.4f}", f"Test IoU: {test_iou:.4f}")

if __name__ == '__main__':
    main()
