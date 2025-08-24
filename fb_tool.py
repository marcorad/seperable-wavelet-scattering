import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import numpy as np

from jws.scattering import filterbank
import matplotlib.pyplot as plt
import numpy as np
from jws.scattering.config import cfg

# Create the figure and axes
ax: plt.Axes
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3)




# ax.set_title("Interactive Line Plot")


# Create text boxes for slopes (m) and intercepts (c)
text_boxes = {}
box_width = 0.1
box_height = 0.05
y_start = 0.15
x_start = 0.15

params = ['d', 'Q', 'alpha', 'alpha_lin', 'alpha_start', 'beta']
params_init = [8, 1, 2.5, 2.5, 2.5, 2.5]

for i in range(3):
    axbox_m = plt.axes([x_start + i * (box_width + 0.25), y_start, box_width, box_height])
    text_box_m = TextBox(axbox_m, params[i*2], initial=str(params_init[i*2]))
    text_boxes[params[i*2]] = text_box_m


    axbox_c = plt.axes([x_start + i * (box_width + 0.25), y_start - 0.08, box_width, box_height])
    text_box_c = TextBox(axbox_c, params[i*2 + 1], initial=str(params_init[i*2 + 1]))
    text_boxes[params[i*2 + 1]] = text_box_c
        
def get_param(p: str):
    return float(text_boxes[p].text)

def plot_fb():
    N = 128
    d = get_param('d')
    Q = get_param('Q')
    alpha = get_param('alpha')
    alpha_lin = get_param('alpha_lin')
    alpha_start = get_param('alpha_start')
    beta = get_param('beta')
    cfg.set_config(
        d=d,Q=Q,
        alpha=alpha,
        alpha_lin=alpha_lin,
        alpha_start=alpha_start,
        beta=beta
    )
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
        ax.plot(np.linspace(-0.5, 0.5, psi.shape[0]), psi, 'k-')
        # ax.plot(np.linspace(-0.5, 0.5, psi.shape[0]), psi_rev, 'r-')
    
    phi = np.fft.fftshift(filterbank.get_wavelet_filter(fb, 0, 0, 1, 0))  
    lp += phi**2
    ax.plot(np.linspace(-0.5, 0.5, phi.shape[0]), phi, 'k--')    
    ax.plot(np.linspace(-0.5, 0.5, lp.shape[0]), lp / np.max(lp), 'k.')    
    
    
    
    ax.set_xlabel("$\omega$")
    ax.set_ylabel("")
    # ax.grid(True)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

# Function to update the plot based on text box values
def submit(text):
    try:
        ax.clear()
        plot_fb()
    except ValueError:
        pass
    fig.canvas.draw_idle()

# Connect the submit function to the text boxes
for box in text_boxes.values():
    box.on_submit(submit)
    
submit('')


plt.show()