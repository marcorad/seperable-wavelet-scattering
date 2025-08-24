from .config import cfg
from .torch_backend import TorchBackend
from .filterbank import scattering_filterbank_separable, get_Lambda_set, get_wavelet_filter, filterbank_to_tensor, calculate_padding_1d, get_output_downsample_factor, calculate_sigma_psi_w, calculate_sigma_phi_w
from typing import List, Tuple, Dict
from torch import Tensor
import torch
from math import ceil
import gc
torch.backends.cuda.cufft_plan_cache[0].max_size = 0


class JointScattering:
    backend: TorchBackend = TorchBackend()
    def __init__(self, N: List[int], d: List[int], Q: List[List[float]], startfreq: List[float] = None, allow_ds: List[bool] = None, remove_highly_corr_filter = False) -> None:
        """Create a separable scattering object, which precalculates the filters.

        Args:
            N (List[int]): The size of each dimension in scattering, corresponding to the input shape (Nbatch, *N).
            d (List[int]): A list of downsampling factors for each dimension, ordered according their appearance in the input.
            Q (List[List[float]]): A list containing the Qs used for each level of the scattering operation. Each item should be a list with Q corresponding to a specific dimension.
            startfreq (List[float], optional): The starting frequencies to place the filters of first-level scattering. When None, the frequency domain is fully covered. Defaults to None.
            allow_ds (bool, optional): Allow downsampling to occur for efficient computation. Defaults to True.
            remove_highly_corr_filters (bool): Whether to remove filters which are highly correlated, i.e., on-axis filters with negative frequency. Defaults to false.    
        """
        self.pad = []
        self.remove_corr_filters = remove_highly_corr_filter
        self.Npad = []
        self.Ndim = len(N)
        self.d = d
        self.Q = Q
        self.Nlevels = len(Q[0])
        if allow_ds == None: allow_ds = [True for _ in range(self.Ndim)]
        self.allow_ds = allow_ds
        assert self.Nlevels <= 3, f'Requested {self.Nlevels} scattering levels. A maximum of 3 levels is supported. More than 3 levels are typically not useful.'
        assert all([d[i] <= N[i] for i in range(self.Ndim)]), f'All invariance scales {d} must be <= the signal support {N}.'
        for i in reversed(range(self.Ndim)):
            l, r, n = calculate_padding_1d(N[i], d[i])
            self.pad.extend([l, r])
            self.Npad.append(n)
        self.Npad.reverse()
            
        self.fb = scattering_filterbank_separable(self.Npad, d, Q, startfreq, allow_ds)
        filterbank_to_tensor(self.fb)
        
    def _mul_and_downsample(self, X: Tensor, level: int, input_ds: List[int], lambdas: List[float]):        
        mul1d = lambda x, y, dim: self.backend.mul1d(x, y, dim+1) #+1 to account for batch dim
        freqds = lambda x, d, dim: self.backend.freq_downsample1d(x, d, dim+1)
        for dim in range(self.Ndim):
            filter = get_wavelet_filter(self.fb, dim, level, input_ds[dim], lambdas[dim])  
            ds = get_output_downsample_factor(self.fb, dim, level, input_ds[dim], lambdas[dim])
            X = mul1d(X, filter, dim)
            if self.allow_ds[dim]: X = freqds(X, ds, dim)
        return X
    
    def _ifft_mul_and_downsample(self, X: Tensor, level: int, input_ds: List[int], lambdas: List[float]):
        mul1d = lambda x, y, dim: self.backend.mul1d(x, y, dim+1) #+1 to account for batch dim
        freqds = lambda x, d, dim: self.backend.freq_downsample1d(x, d, dim+1)
        ifft1d = lambda x, dim: self.backend.ifft1d(x, dim)
        for dim in range(self.Ndim):
            filter = get_wavelet_filter(self.fb, dim, level, input_ds[dim], lambdas[dim])  
            ds = get_output_downsample_factor(self.fb, dim, level, input_ds[dim], lambdas[dim])
            X = mul1d(X, filter, dim)
            if self.allow_ds[dim]: X = freqds(X, ds, dim)
            X = ifft1d(X, dim)
        return X
    
    def _ifft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.ifft(X, dims)
    
    def _fft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.fft(X, dims)
    
    def _get_compounded_downsample_factor(self, level, current_ds, lambdas):
        ds = [1 for _ in range(self.Ndim)]
        for dim in range(self.Ndim):
            if self.allow_ds[dim]: ds[dim] = get_output_downsample_factor(self.fb, dim, level, current_ds[dim], lambdas[dim]) * current_ds[dim]
        return ds
    
    def _should_prune(self, lambda_filt: List[float], lambda_demod: List[float], level: int):
        for dim, (lf, ld) in enumerate(zip(lambda_filt, lambda_demod)):
            beta = cfg.get_beta(self.d[dim],self.Q[dim][level])
            sigma_psi_w_demod = max(calculate_sigma_psi_w(self.d[dim], self.Q[dim][level-1]) * abs(ld), calculate_sigma_phi_w(self.d[dim], self.Q[dim][level-1]))
            sigma_psi_w_filt = max(calculate_sigma_psi_w(self.d[dim], self.Q[dim][level-1]) * abs(lf), calculate_sigma_phi_w(self.d[dim], self.Q[dim][level-1]))
            # prune only when demodulated filter's centre freq is not at least within beta standard deviations of the current filter
            # note that we use the abs value of the lambdas since lambdas can be positive or negative
            # |----------*----------|
            # |-----ddddd*ddddd-----|
            #            <---->          
            # |----------*----x-----|
            #               <->       
            # these intervals must overlap, 
            # * is the centre of the spectrum, d is the significant bandwidth of the demodulated filter (via modulus), and f is the morlet filter under consideration which has a center x
            EPS = 1e-9 #for floating point error
            if cfg.get_beta_prune(self.d[dim],self.Q[dim][level])*sigma_psi_w_demod < abs(lf): return True # - sigma_psi_w_filt * beta + EPS
        return False
    
    def scattering(self, x: Tensor, normalise = False, batch_size = None, scat_to_cpu=True) -> Tensor:
        """Perform a separable scattering transform on a real signal x.

        Args:
            x (Tensor): A tensor with shape (Nbatch, ...), the dimensions from 1 onwards are the scattering dimensions.
            normalise (bool, optional): normalise scattering coefficients with respect to the previous level. Defaults to False.
            batch_size (int, optional): compute scattering in batches for memory usage reduction. If None, computes all items in the batch dimension. Default to None.
            
        Returns:
            Tensor: The scattering features, with the last axis corresponding to the various filters paths.
        """
        N = x.shape[0]
        if batch_size == None: batch_size = x.shape[0]
        x = x.cpu()
        S = []
        for i in range(ceil(N / batch_size)):   
            begin = i * batch_size
            end = min((i+1)*batch_size, N)
            x_b = x[begin:end, ...].to(cfg.DEVICE)        
            s, _, _ = self._scattering(x_b, returnU=False, returnSpath=False, normalise=normalise, scat_to_cpu=scat_to_cpu) 
            S.append(self.backend.stack(s, dim=1)) # stack all features into a single tensor              
        return self.backend.concat(S, dim=0) # concat all batches
    
    def _calculate_paths(self):   #TODO: use for pre-calculating separable filters     
        paths = []                
        lambda_zero = tuple([0 for _ in range(self.Ndim)]) #tuple of zeros for the LPF (phi)
        l0_compounded_ds = [1 for _ in range(self.Ndim)] #no downsampling on the input
        paths.append({
            'level': 0,
            'lambda': (lambda_zero,),
            'phi_ds': l0_compounded_ds
        })        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, [1]*self.Ndim, self.remove_corr_filters)
        for lambda1 in Lambda_1:  
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            paths.append({
                'level': 1,
                'lambda': (lambda1,),
                'psi_ds': l0_compounded_ds,
                'phi_ds': l1_compounded_ds,
            })            
            if self.Nlevels == 1: continue            
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds, self.remove_corr_filters)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1, 1): continue #prune the paths, since downsampling prunes to an inexact extent
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                paths.append({
                    'level': 2,
                    'lambda': (lambda1, lambda2),
                    'psi_ds': l1_compounded_ds,
                    'phi_ds': l2_compounded_ds
                })              
                                                                       
        return paths
    
    def _normalise(self, x1: Tensor, xn: Tensor):
        EPS = 1e-10
        return x1 / (xn + EPS)
        
    def _scattering(self, x: Tensor, returnU = False, returnSpath = False, normalise=False, scat_to_cpu = True):      
        
        # Kymatio's scattering has a near-identical implementation
          
        #function aliases for clarity        
        #TODO: better unpadding
        unpad = lambda x: self.backend.unpad(x) if all(self.allow_ds) else x #disable unpadding when DS occurs
        pad = lambda x, s: self.backend.pad(x, s)
        fft = lambda x: self._fft_all(x)
        ifft = lambda x: self._ifft_all(x)        
        mulds = lambda x, level, ids, lambdas: self._mul_and_downsample(x, level, ids, lambdas)
        modulus = lambda x: self.backend.modulus(x)        
        ifftmulds = lambda x, level, ids, lambdas: self._ifft_mul_and_downsample(x, level, ids, lambdas)
        
        #pad the tensor
        x = pad(x, self.pad)
        #get the fft of the input signal across all dimensions
        lambda_zero = tuple([0 for _ in range(self.Ndim)]) #tuple of zeros for the LPF (phi)
        X = fft(x)
        del x
        s_0 = unpad(ifft(mulds(X, 0, [1 for _ in range(self.Ndim)], lambda_zero)).real)
        S = [s_0.cpu() if scat_to_cpu else s_0]
        Up = {}
        Sp = {}
        
        l0_compounded_ds = [1 for _ in range(self.Ndim)] #no downsampling on the input
        
        if returnSpath: Sp[lambda_zero] = s_0
        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, [1]*self.Ndim, self.remove_corr_filters)
        for lambda1 in Lambda_1:   
            # if scat_to_cpu: gc.collect()
            torch.cuda.empty_cache()
            u_1 = modulus(ifft(mulds(X, 0, l0_compounded_ds, lambda1)))
            U_1 = fft(u_1)
            if not returnU: del u_1
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            s_1 = ifft(mulds(U_1, 0, l1_compounded_ds, lambda_zero)).real
            s_1 = unpad(s_1)
            if normalise: s_1 = self._normalise(s_1, s_0)
            if scat_to_cpu:                 
                S.append(s_1.cpu())
                del s_1
            else: S.append(s_1)   
                     
            
            if returnU:     Up[lambda1] = u_1 
            if returnSpath: Sp[lambda1] = s_1
            
            if self.Nlevels == 1: 
                del U_1
                continue
            
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds, self.remove_corr_filters)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1, 1): continue #prune the paths, since downsampling prunes to an inexact extent
                # print(f'\t{lambda2}')
                u_2 = modulus(ifft(mulds(U_1, 1, l1_compounded_ds, lambda2)))
                U_2 = fft(u_2)
                if not returnU: del u_2
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                s_2 = unpad(ifft(mulds(U_2, 1, l2_compounded_ds, lambda_zero)).real)
                if normalise: s_2 = self._normalise(s_2, s_1)
                if scat_to_cpu:                 
                    S.append(s_2.cpu())
                    del s_2
                else: S.append(s_2)     
                
                
                if returnU:     Up[(lambda1, lambda2)] = u_2
                if returnSpath: Sp[(lambda1, lambda2)] = s_2
                del U_2
            del U_1
            
                
                    
        return S, Sp, Up
        
        
        
        