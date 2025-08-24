from jws.scattering import filterbank
import matplotlib.pyplot as plt
import numpy as np
from jws.scattering.config import cfg

Q = 1
cfg.set_alpha(Q,    2.5, False)
cfg.set_alpha(Q,    2.0, True)
cfg.set_beta(Q,     2.5)
cfg.NORMALISE_LITTLE_WOOD_PALEY = True
cfg.FORCE_ANALYTICITY = True

N = [256, 256]
d = [8,8]
Npad = [filterbank.calculate_padding_1d(n, di)[2] for n, di in zip(N, d)]
print(Npad)

fb = filterbank.scattering_filterbank_separable(Npad, d, [[Q], [Q]], allow_ds=[False, False])
lambdas = filterbank.get_Lambda_set(fb, 0, [1, 1])
print(len(lambdas))
for l in lambdas:
    print(f'{l[0]:.3f}, {l[1]:.3f}')


fig = plt.figure()
ax = fig.add_subplot(111)
X = np.linspace(-0.5, 0.5, Npad[1])
Y = np.linspace(-0.5, 0.5, Npad[0])

s = np.zeros(Npad)


c = 0.85

for l in lambdas:
    psi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0])
    psi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, l[1])
    psi = psi0[:, None] * psi1[None, :] # broadcast for getting wavelet
    s += psi**2
    print(np.max(psi))     
    ax.contour(X, Y, np.fft.fftshift(psi), [c], colors='k', linewidths=1.0)
    
    
phi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, 0)
phi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, 0)
phi = phi0[:, None] * phi1[None, :] # broadcast for getting wavelet
plt.contour(X, Y, np.fft.fftshift(phi), [c], colors='k', linestyles='dashed', linewidths=1.0)

ax.axvline(c='k', lw=1)
ax.axhline(c='k', lw=1)
ax.axis('off')
    
plt.ylim([-0.05, 0.5])

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')

# x, y = np.meshgrid(X, Y)
# ax.plot_surface(x, y, np.fft.fftshift(s))

ax.text(0.55, 0.0, '$\omega^{(1)}$', horizontalalignment='center', verticalalignment='center', parse_math=True)
ax.text(0.0, 0.53, '$\omega^{(2)}$', horizontalalignment='center', verticalalignment='center', parse_math=True)

plt.show()
    
    