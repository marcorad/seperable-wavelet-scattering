from scipy.special import lambertw as W
from jws.scattering.morlet import sample_gauss, morlet_filter_freq
import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Carlito']})
# rc('text', usetex=True)

eps = 0.001

Q = 1
alpha = 3
sigma = 1/alpha * (2**(1/Q)-1)
print(f'{sigma=}')
g_sup = sample_gauss(0, 1/sigma)
gamma = sample_gauss(-1, sigma) / sample_gauss(0, sigma)
print(f'{g_sup=}, {gamma=}, {gamma/g_sup=}')

N = 64
s = 0.125
m1 = morlet_filter_freq(N, s*np.pi, 1/sigma)
m2 = morlet_filter_freq(N, s*np.pi * (2**1/Q), 1/sigma)
m3 = morlet_filter_freq(N, s*np.pi * (2**2/Q), 1/sigma)
zero = True
if zero:
    m1[N//2+1:] = 0
    m2[N//2+1:] = 0
    m3[N//2+1:] = 0
# m4 = morlet_filter_freq(1000, 0.1*3.14 * (2**3/Q), 1/sigma)[0:500]
lp = m1**2 + m2**2 + m3**2
normalise = True
if normalise:
    n = np.max(lp)**0.5
    m1 = m1/n
    m2 = m2/n
    m3 = m3/n
    # m4 = m4/n
    lp = m1**2 + m2**2 + m3**2 
plt.figure()
plt.subplot(311)
plt.plot(m1)
plt.plot(m2)
plt.plot(m3)
plt.plot(lp)
plt.subplot(312)
m3t = fftshift(ifft(m3))
plt.plot(np.real(m3t))
plt.plot(np.imag(m3t))
plt.plot(np.abs(m3t))
plt.subplot(313)
m1t = fftshift(ifft(m1))
plt.plot(np.real(m1t))
plt.plot(np.imag(m1t))
plt.plot(np.abs(m1t))
# plt.show(block=True)

Q = np.linspace(0.5, 2, 100)
alpha = np.linspace(1, 5, 80)
qi, ai = np.meshgrid(Q, alpha)

sigma = 1/ai * (2**(1/qi)-1)
g_sup = sample_gauss(0, 1/sigma)
gamma = sample_gauss(-1, sigma) / sample_gauss(0, sigma)
eps = gamma/g_sup
eps_log = np.log(eps)

plt.figure()
CS = plt.contour(Q, alpha, eps, levels=[0.0001, 0.001, 0.01, 0.1, 0.25], colors='k', linewidths = 1.0)

def fmt(x):
    return f'$\\epsilon={x}$'
plt.clabel(CS, inline=1, fontsize=10, fmt=fmt)



# plt.gca().set_xscale('log')
plt.xlabel('$Q$')
plt.ylabel('$\\alpha$')
# eps = 0.2
# aw = (2**(1/Q) - 1) / (W((eps**2) / np.pi / 2)**0.5) / 9
# plt.plot(Q, aw)
plt.show()


