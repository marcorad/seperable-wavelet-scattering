import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import numpy as np
from typing import List

from jws.scattering import filterbank
import matplotlib.pyplot as plt
import numpy as np
from jws.scattering.config import cfg

def plot_fb(d, Q, ax: plt.Axes):
    N = 256
    cfg.FORCE_ANALYTICITY = True
    cfg.NORMALISE_LITTLE_WOOD_PALEY = False
    
    Npad = int(filterbank.calculate_padding_1d(N, d)[2])
    print(f'{d=}, {Q=}, {Npad=}')
    
    fb = filterbank.scattering_filterbank_separable([Npad], [d], [[Q]], allow_ds=[False])
    lambdas = filterbank.get_Lambda_set(fb, 0, [1])
    # print(lambdas)
    lp = 0
    
    for l in lambdas:
        psi = np.fft.fftshift(filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0]))    
        # psi_rev = np.flip(psi)   
        lp += psi**2
        ax.plot(np.linspace(-0.5, 0.5, psi.shape[0]), psi, 'k-', linewidth=0.75)
        # ax.plot(np.linspace(-0.5, 0.5, psi.shape[0]), psi_rev, 'r-')
    
    phi = np.fft.fftshift(filterbank.get_wavelet_filter(fb, 0, 0, 1, 0))  
    lp += phi**2
    ax.plot(np.linspace(-0.5, 0.5, phi.shape[0]), phi, 'k--', linewidth=0.75)    
    # ax.plot(np.linspace(-0.5, 0.5, lp.shape[0]), lp / np.max(lp), 'k.')    
    
    
    
    # ax.set_xlabel("$\omega$")
    ax.set_ylabel("")
    # ax.grid(True)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_yticklabels('')
    
    alpha = cfg.get_alpha(d, Q)
    alpha_start = cfg.get_alpha_start(d, Q)
    beta = cfg.get_beta(d, Q)
    start = '\\text{start}'
    fmt = {'fontsize': 8}
    ax.text(0.39, 0.5, f"$\\alpha={alpha}$", fmt)
    ax.text(0.39, 0.3, f"$\\alpha_s={alpha_start}$", fmt)
    ax.text(0.39, 0.1, f"$\\beta={beta}$", fmt)
    
ds = [2, 4, 6, 8]
Qs = [1]

axs: List[List[plt.Axes]]
fig, axs = plt.subplots(4, 1)

for col, Q in enumerate(Qs):
    for row, d in enumerate(ds):
        plot_fb(d, Q, axs[row][col])
        
for r in range(4):
    axs[r][0].set_ylabel(f'$d={ds[r]}$')
    
axs[0][0].set_title('$Q = 0.75$', {'fontsize': 10})
# axs[0][1].set_title('$Q = 1$', {'fontsize': 10})
    
plt.show(block=True)