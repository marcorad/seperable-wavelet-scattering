from jws.scattering import filterbank
import matplotlib.pyplot as plt
import numpy as np
from jws.scattering.config import cfg

def normalise_fb(fb):
    lambdas = filterbank.get_Lambda_set(fb, 0, [1,1])
    n_factors = {}
    
    # normalise energies for off-axis filters
    e = 0
    for l in lambdas:
        if l[0] == 0 or l[1] == 0: continue
        psi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0])
        psi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, l[1])
        psi = psi0[:, None] * psi1[None, :] # broadcast for getting wavelet
        # print(l, np.max(psi**2))
        e += psi**2
    print(np.max(e))
    n = np.max(e)
    for l in lambdas:
        if l[0] == 0 or l[1] == 0: continue
        assert l not in n_factors.keys()
        n_factors[l] = n
        
    # normalise energies for on-axis filters
    e = 0
    for l in lambdas:
        if not (l[0] == 0 or l[1] == 0): continue
        psi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0])
        psi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, l[1])
        psi = psi0[:, None] * psi1[None, :] # broadcast for getting wavelet
        # print(l, np.max(psi**2))
        e += psi**2
    n = np.max(e)
    print(np.max(e))
    for l in lambdas:
        if not (l[0] == 0 or l[1] == 0): continue
        assert l not in n_factors.keys()
        n_factors[l] = n
    n_factors[(0, 0)] = 1
        
    return n_factors


def lpsum(fb, n_factors):
    lambdas = filterbank.get_Lambda_set(fb, 0, [1,1])
    s = 0
    # add the filters
    for l in lambdas:
        psi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0])
        psi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, l[1])
        psi = psi0[:, None] * psi1[None, :]# broadcast for getting wavelet
        # print(l, np.max(psi**2))
        s += psi**2  / n_factors[l] 
        
    # s = s*2 # account for filter analyticity
        
    # add the lpf
    phi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, 0)
    phi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, 0)
    phi = phi0 [:, None]* phi1[None, :]
    # print(np.max(phi**2))
    s += phi**2
    
    return np.fft.fftshift(s[:, :])

Q = 0.75
cfg.set_alpha(Q,    2.5, False)
cfg.set_alpha(Q,    2.3, True)
cfg.set_beta(Q,     2.5)
cfg.NORMALISE_LITTLE_WOOD_PALEY = False
cfg.FORCE_ANALYTICITY = True

N = [256, 256]
d = [8]*2
Npad = [filterbank.calculate_padding_1d(n, di)[2] for n, di in zip(N, d)]
print(Npad)

fb = filterbank.scattering_filterbank_separable(Npad, d, [[Q], [Q]], allow_ds=[False, False])
n_factors = normalise_fb(fb)
energy = lpsum(fb, n_factors)

energy_norm = energy / np.max(energy)
print(np.min(energy_norm), np.max(energy_norm))

plt.imshow(energy_norm, origin='lower', vmax=1.0)
plt.show(block=True)
