"""Load released PUResNet v1 Keras weights (whole_trained_model1.hdf) into PUResNetV1Faithful3D.

Usage: python scripts/import_puresnet_keras_weights.py <keras.hdf> <out.pth>
Keras Conv3D kernels (kx, ky, kz, cin, cout) become PyTorch (cout, cin, kx, ky, kz); channels_last input
(x, y, z, c) maps to (c, x, y, z). BatchNorm gamma/beta/moving_mean/moving_variance map to weight/bias/running_*.
"""
import sys
import h5py
import torch

sys.path.insert(0, ".")
from models.LiteratureModels3D import PUResNetV1Faithful3D

BN = {"gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}


def load_keras_puresnet(model, h5_path):
    state = model.state_dict()
    arrays = {}
    with h5py.File(h5_path, "r") as handle:
        root = handle["model_weights"] if "model_weights" in handle else handle
        root.visititems(lambda name, obj: arrays.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
    loaded = set()
    for name, array in arrays.items():
        layer, weight = name.split("/")[-2], name.split("/")[-1].split(":")[0]
        tensor = torch.from_numpy(array)
        if layer.startswith("bn"):
            key = f"L.{layer}.{BN[weight]}"
        elif weight == "kernel":
            key, tensor = f"L.{layer}.weight", tensor.permute(4, 3, 0, 1, 2)
        else:
            key = f"L.{layer}.bias"
        assert state[key].shape == tensor.shape, (key, state[key].shape, tensor.shape)
        state[key].copy_(tensor)
        loaded.add(key)
    missing = [k for k in state if k not in loaded and not k.endswith("num_batches_tracked")]
    assert not missing, missing[:10]
    model.load_state_dict(state)
    return len(loaded)


if __name__ == "__main__":
    net = PUResNetV1Faithful3D(18, 1, 18)
    print("tensors loaded:", load_keras_puresnet(net, sys.argv[1]))
    torch.save(net.state_dict(), sys.argv[2])
