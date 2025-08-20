import numpy as np
import time
import pickle

from VIBinMM import VIBinMM
from utils import plot_sphere_data


with open('data/syntheticdata.pkl', 'rb') as f:
    syntheticdata = pickle.load(f)

X = syntheticdata['X']
labels = syntheticdata['y']

plot_sphere_data(X, labels, 'Orginal Synthetic Data')

n_components = np.size(np.unique(labels))
N, D = X.shape


print('begin training......')
print('========================dataset is {}========================'.format('Synthtic Bingham MM'))

Zs = []
Ms = []
Wgt = []

n = 0
while n < 10:

    print(f'================ Loop {n} ======================')

    vbmm = VIBinMM(n_components=n_components, max_iter=1500, thred=4e-2, init_params='random_from_data')

    begin = time.time()
    vbmm.fit(X, verbose=0)
    end = time.time()

    print("time: {}".format(end - begin))
    n += 1

    Zs.append(vbmm.Z_bhm)
    Ms.append(vbmm.M_bhm)
    Wgt.append(vbmm.weight_bhm)

Zs = np.average(np.asarray(Zs), axis=0)
Ms = np.average(np.asarray(Ms), axis=0)
Wgt = np.average(np.asarray(Wgt), axis=0)

print('Zs: ', Zs)
print('Ms: ', Ms)
print('Wgt: ', Wgt)

X_synth, y_synth = vbmm.rvs(N)
plot_sphere_data(X_synth, y_synth, 'Sampling Synthetic Data')









