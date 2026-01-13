import h5py
import numpy as np

f = h5py.File('/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set_cache/box72/1a1e.h5', 'r')
mask = f['label/binding_site'][:]
print('Toplam 1:', np.sum(mask > 0.5))
print('Grid shape:', mask.shape)
print('Pozitif oran:', np.sum(mask > 0.5) / mask.size)