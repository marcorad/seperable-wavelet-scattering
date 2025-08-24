from sympy import divisors
from typing import Dict, List
import numpy as np
from .config import cfg
from .morlet import morlet_filter_freq, gauss_filter_freq
from math import floor
import torch
from torch import Tensor
from itertools import product

def calculate_padding_1d(N: int, d: int):
    assert d % 2 == 0, 'd must be even.'
    left = d//2
    c = d - (N % d)
    right = left + c
    N_pad = N + left + right
    return left, right, N_pad

def calculate_sigma_psi_w(d, Q):
    alpha = cfg.get_alpha(d, Q)
    return 1 / alpha * (2**(1/Q) - 1)

def calculate_sigma_phi_w(d, Q):
    sigma_phi = d * cfg.get_beta(d, Q) / np.pi # phi time std
    sigma_phi_w = 1 / sigma_phi # phi freq std
    return sigma_phi_w
    

def _filterbank_1d(N: int, d: int, Q: float, startfreq: float = None, include_negative_lambdas = True, input_ds_factors: List[int] = None) -> Dict: 
    """
    Generate a 1d Morlet scattering filterbank sampled at all possible sampling frequencies required for subsequent scattering operations.

    Args:
        N (int): Signal length (already padded)
        d (int): Invariance scale in samples, specified as the downsampling factor
        Q (float): Number of filters per octave
        startfreq (float, optional): Normalised frequency in [0, 0.5] (i.e., frequency / sampling_frequency) at which the first filter starts. When None, the first filter is placed according filterbank parameters alpha and beta. Defaults to None. 
        include_negative_lambdas (bool, optional): Include negative frequencies in the filterbank. This is only necessary for all dimensions after the 1st dimension. Defaults to True.
        input_ds_factors (List[int]): A list of input downsampling factors to compute. When None, computes all required combinations of input and output downsampling factors. Defaults to None. Should be used to prune the list

    Returns:
        Dict: A dictionary containing all information required to perform calculations with this filterbank.
        List: a list of all the unique compounded donwsampling factors of psi (d_in * d_out) to be used by further filterbanks in deeper scattering levels.
        
    How to use the filterbank:        
        phi: fb[amount of input downsampling]['phi'] -> filter;          
        psi: fb[amount of input downsampling]['psi'][lambda] -> filter;         
        output downsampling amount: fb[amount of input downsampling]['ds'][lambda] -> amount to downsample after filtering (if lambda=0, then downsampling for the lpf).        
    """
    
    assert N % d == 0, f'Signal must be padded such that d|N ({d} does not divide {N})'
    # all the possible compounded downsampling steps that can be performed on a single filter, 
    # this can be pruned later according to the needs of all FBs, which this function is agnostic to
      
    if input_ds_factors == None: input_ds_factors = [1]
    
    alpha = cfg.get_alpha(d, Q)
    alpha_lin = cfg.get_alpha_lin(d, Q)
    alpha_start = cfg.get_alpha_start(d, Q)
    beta = cfg.get_beta(d, Q)
    pi = np.pi
    
    sigma_phi = d * beta / pi # phi time std
    sigma_phi_w = 1 / sigma_phi # phi freq std
    
    sigma_psi_w = calculate_sigma_psi_w(d,Q) # psi freq std
    sigma_psi = 1 / sigma_psi_w # psi time std
    
    lambda_0 = startfreq * 2 * pi if startfreq else sigma_phi_w * alpha_start    
    
    # assert lambda_0 >= sigma_phi_w * alpha_start, f'A starting frequency of {startfreq} is too small. Must be at least {sigma_phi_w * alpha_lin / 2 / pi}.'
    # assert lambda_0 < pi, f'Invariance scale to small for parameters alpha={alpha_lin}, beta={cfg.get_beta(d, Q)} to allow for a single filter. Try increasing the invariance scale or decreasing beta and/or alpha'
    
    fb = {}    
    compounded_output_ds_factors = set()
    for d_i in input_ds_factors:
        morlet_params = []
        lambda_ = lambda_0
        upper_limit = pi / d_i
        #get all linear lambdas
        while lambda_*sigma_psi_w < sigma_phi_w and lambda_ < upper_limit:
            morlet_params.append((lambda_, sigma_phi * lambda_)) #limit the bandwidth to time support
            if (lambda_*2**(1/Q))*sigma_psi_w < sigma_phi_w:
                lambda_ += alpha_lin * sigma_phi_w #place the next filter alpha_lin stds away if we can't scale exponentially yet
            else:
                lambda_ *= 2**(1/Q) # we can place exponentially - break out
                break
        #get all exponential lambdas
        while lambda_ <= upper_limit:
            morlet_params.append((lambda_, sigma_psi))
            lambda_ *= 2**(1/Q) #sigma_psi is constructed such that the current wavelet at lambda_ decays by alpha stds at the wavelet placed at lambda_*2^(1/Q) 
            
        #include negative lambdas if required
        if include_negative_lambdas:
            neg_params = [(-p[0], p[1]) for p in morlet_params]
            neg_params.reverse()
            neg_params.extend(morlet_params)
            morlet_params = neg_params #morlet parameters is sorted from smallest to largest lambda
            
        if len(morlet_params) == 0: print(f'NOTE: no scattering filterbank exists for input downsampling of {d_i} and invariance scale {d}. All required information is retained in the previous filterbank.')
            
        # sample all morlets
        morlets = {}
        for lambda_, sigma_t in morlet_params:
            morlets[lambda_] = morlet_filter_freq(N//d_i, lambda_=lambda_*d_i, sigma=sigma_t, force_analyticity=cfg.FORCE_ANALYTICITY, norm_peak_one=cfg.NORMALISE_PEAK_ONE) # since we have an input downsampling by d_i, we must scale lambda accordingly
            
        if cfg.NORMALISE_LITTLE_WOOD_PALEY:
            esum = np.zeros(N//d_i)
            for l, m in morlets.items():
                esum += m**2
                
            norm = np.max(esum)**0.5
            for l in morlets.keys():
                morlets[l] = morlets[l] / norm
            
        #compute the amount each morlet may be downsampled
        ds_after_filtering = {}
        for p in morlet_params:
            sigma_t = min(p[1]/p[0], sigma_phi)  
            EPSILON = 1e-9 # some error margin allowed
            ds = max(floor(pi * sigma_t / beta / d_i + EPSILON), 1) # this becomes pi / ( lambda * (2^(1/Q) - 1) * beta / alpha ) / d_i
            while d % ds != 0: ds -= 1
            while d % (ds * d_i) != 0: ds -= 1
            ds_after_filtering[p[0]] = ds
            compounded_output_ds_factors.add(ds * d_i)
            assert d % (ds * d_i) == 0
        ds_after_filtering[0] = d // d_i
            
        # sample the gaussian lpf
        gauss = gauss_filter_freq(N//d_i, sigma_phi / d_i) # since we have an input downsampling by d_i, we must scale sigma accordingly
        
        fb[d_i] = {
            'phi': gauss,
            'psi': morlets,
            'ds': ds_after_filtering
        }        
                
        #wavelets will also need access to lpf after they have been downsampled
        #add LPF for all the combinations of the output ds, since we will use the LPF in this level combined with psi output
        for d_o in compounded_output_ds_factors:
            if d_o not in fb.keys(): #only add if needed, since some morlets may already store this LPF
                gauss = gauss_filter_freq(N//d_o, sigma_phi / d_o)
                fb[d_o] = {
                    'phi': gauss,
                    'psi': None, #no psi is required for this level
                    'ds': {0: d // d_o}
                }                
        
                
        #some wavelets also directly use the LPF, so add the direct LPF
        if d not in fb.keys(): 
            gauss = gauss_filter_freq(N//d, sigma_phi / d)
            fb[d] = {
                'phi': gauss,
                'psi': None,
                'ds': {0: 1}
            }
        compounded_output_ds_factors.add(d)
    
    return fb, list(compounded_output_ds_factors)


def _scattering_filterbanks_1d(N, d, Q: List[float], startfreq: float = None, allow_ds = True, include_neg_lambdas = False):
    """Generate multi-dimensional separable filterbanks for a maximum level of scattering defined by the length of Q.

    Args:
        N (List[int]):  List of signal lengths (already padded) for each dimension
        d (List[int]): List of invariance scales for each dimension
        Q (List[List[float]]): List of filter per octaves for each level and each dimension, e.g. [[2, 2], [1, 1]] for a 2-level transform with for Q = (2, 2) for level 1 and Q = (1, 1) for level 2.
        startfreq (List[float], optional): List of normalised start frequencies in [0, 0.5] (frequency/sampling_frequency) for each dimension. When None, the first filter is placed according to filterbank parameters. Defaults to None.
        allow_ds (bool, optional): Allows downsampling to occur. Note that padding operation will not operate properly when False. Defaults to True.

    Returns:
        List: A list containing a filterbank dictionary for each scattering level and dimension.
        
    How to use the filterbanks:        
        phi: fb[level][amount of input downsampling]['phi'] -> filter;          
        psi: fb[level][amount of input downsampling]['psi'][lambda] -> filter;         
        output downsampling amount: fb[level][amount of input downsampling]['ds'][lambda] -> amount to downsample after filtering (if lambda=0, then downsampling for the lpf). 
    """
    FB = []
    ds = [1] #first filterbank has no input downsampling
    for j, q in enumerate(Q):
        fb, ds = _filterbank_1d(N, d, q, startfreq if j == 0 else None, input_ds_factors=ds, include_negative_lambdas=include_neg_lambdas)
        if not allow_ds: ds = [1] 
        FB.append(fb)
    return FB

def scattering_filterbank_separable(N: List[int], d: List[int], Q: List[List[float]], startfreq: List[float] = None, allow_ds: List[bool] = None):
    if startfreq == None: startfreq = [None for _ in range(len(N))]
    FB = []
    if allow_ds == None: allow_ds = [False for _ in range(len(N))]
    for i, (N_i, d_i, Q_i, startfreq_i) in enumerate(zip(N, d, Q, startfreq)):
        FB.append(_scattering_filterbanks_1d(N_i, d_i, Q_i, startfreq_i, allow_ds[i], include_neg_lambdas=i>0)) #fb[dim][level][amount of input downsampling]['phi'/'psi'/'ds']<[lambda for 'psi']>
    return FB

def get_wavelet_filter(fb: List, dim: int, level: int, input_ds: int, lambda_: float): 
    """Helper function to get a morlet filter from a filterbank created with scattering_filterbanks. Returns the LPF phi if lambda = 0.

    Args:
        fb (List): The filterbank generated with scattering_filterbanks
        dim (int): The dimension index
        level (int): The filterbank level
        input_ds (int): The compounded amount of input downsampling applied
        lambda_ (float): The filter frequency. If 0, returns the LPF.

    Returns:
        The wavelet, which can either be a numpy array or torch.tensor
    """
    if lambda_ == 0: return fb[dim][level][input_ds]['phi']
    return fb[dim][level][input_ds]['psi'][lambda_]

def normalise_wavelet_filter(fb: List, dim: int, level: int, input_ds: int, lambda_: float, norm: float): 
    if lambda_ == 0: return
    fb[dim][level][input_ds]['psi'][lambda_] /= norm

def get_output_downsample_factor(fb: List, dim: int, level: int, input_ds: int, lambda_: float) -> int:
    """Helper function to get the amount of downsampling required after performing filtering.

    Args:
        fb (List): The filterbank generated with scattering_filterbanks
        level (int): The filterbank level
        input_ds (int): The compounded amount of input downsampling applied
        lambda_ (float): The filter frequency. If 0, returns the LPF downsample amount.

    Returns:
        int: The downsampling amount required after filtering.
    """
    return fb[dim][level][input_ds]['ds'][lambda_]

def _for_each_filter(fb, func):
    """Perform a function on each filter in a scattering filterbank.

    Args:
        fb: The filterbank
        func: The function to perform on the filter before storing it in the filterbank.
    """
    Nlevels = len(fb[0])
    Ndims = len(fb)
    for dim in range(Ndims):
        for level in range(Nlevels):
            for ids in fb[dim][level].keys():
                #phi
                fb_dim = fb[dim]
                phi = get_wavelet_filter(fb, dim, level, ids, 0)                
                fb_dim[level][ids]['phi'] = func(phi)
                #psi
                if fb_dim[level][ids]['psi'] != None: #there may be some LPFs stored which do not require psi
                    for l in fb_dim[level][ids]['psi'].keys(): #for all lambdas
                        psi = fb_dim[level][ids]['psi'][l]
                        fb_dim[level][ids]['psi'][l] = func(psi)
                    
def _convert_to_tensor(filt):
    """Convert an input to a torch tensor on the configured device in config.cfg.

    Args:
        filt: The filter (torch.Tensor or numpy.ndarray)

    Returns:
        Tensor: The converted input.
    """
    if type(filt) == np.ndarray:
        return torch.from_numpy(filt).type(cfg.REAL_DTYPE).to(cfg.DEVICE)
    if filt == None: return None
    raise Exception('Must be a numpy array or None!')
    

def filterbank_to_tensor(fb: List):
    """Convert all filters to Tensors on the configured device in config.cfg.

    Args:
        fb (List): The filterbank generated with scattering_filterbanks.
    """
    _for_each_filter(fb, _convert_to_tensor)
    
def get_Lambda_set(fb: List, level: int, input_ds, remove_highly_corr_filters = False):
    """Get the set of all lambda filter combinations for a filterbank given a level and amount of compounded input downsampling.

    Args:
        fb (List): The filterbank generated with scattering_filterbanks.
        level (int): The scattering level (filterbank index)
        input_ds (int): The compounded amount of input downsampling
        remove_highly_corr_filters (bool): Whether to remove filters which are highly correlated, i.e., on-axis filters with negative frequency
    Returns:
        List[Tuple[float...]]: A list containing tuples of all centre frequencies of a multidimensional separable filter.
    """
    lambdas = []
    for dim, fb_dim in enumerate(fb):
        lambdas.append([0] + list(fb_dim[level][input_ds[dim]]['psi'].keys()))
    Lambda = list(product(*lambdas))[1:] # remove vec(0) lambda
    if remove_highly_corr_filters:
        Lambda_final = []
        for L in Lambda:
            if any([l == 0 for l in L]) and any([l <0 for l in L]):
                continue
            Lambda_final.append(L)
        return Lambda_final
    else:
        return Lambda


        

    