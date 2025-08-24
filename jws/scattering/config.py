import torch
from typing import Dict, List, Literal, Tuple

# default parameters
ALPHA_DEFAULT = 2.0
ALPHA_LIN_DEFAULT = 2.0
ALPHA_START_DEFAULT = 2.0
BETA_DEFAULT = 2.5
BETA_PRUNE_DEFAULT = 1.0

class Config:         
    
    
    def __init__(self):        
        self.config_dict: Dict[Tuple[int, int], Dict[str, float]] = {} # (d, Q) -> alpha; alpha_lin; alpha_start; beta; beta_prune
        # options
        self.FORCE_ANALYTICITY = True
        self.NORMALISE_LITTLE_WOOD_PALEY = False
        self.NORMALISE_PEAK_ONE = True
        self.DEVICE: torch.device = torch.device('cpu')
        self.COMPLEX_DTYPE: torch.dtype = torch.complex64
        self.REAL_DTYPE: torch.dtype = torch.float32 
    
    def set_config(self, d, Q, 
                   alpha=ALPHA_DEFAULT, 
                   alpha_lin=ALPHA_LIN_DEFAULT, 
                   alpha_start=ALPHA_START_DEFAULT, 
                   beta=BETA_DEFAULT, 
                   beta_prune=BETA_PRUNE_DEFAULT):
        self.config_dict[(d, Q)] = {
            'alpha'         : alpha,
            'alpha_lin'     : alpha_lin,
            'alpha_start'   : alpha_start,
            'beta'          : beta,
            'beta_prune'    : beta_prune,
        }
        
    def get_alpha(self, d, Q):
        Q = float(Q)
        if (d, Q) in self.config_dict.keys(): return self.config_dict[(d,Q)]['alpha']
        else: return ALPHA_DEFAULT
        
    def get_alpha_lin(self, d, Q):
        Q = float(Q)
        if (d, Q) in self.config_dict.keys(): return self.config_dict[(d,Q)]['alpha_lin']
        else: return ALPHA_LIN_DEFAULT
        
    def get_alpha_start(self, d, Q):
        Q = float(Q)
        if (d, Q) in self.config_dict.keys(): return self.config_dict[(d,Q)]['alpha_start']
        else: return ALPHA_START_DEFAULT
        
    def get_beta(self, d, Q):
        Q = float(Q)
        if (d, Q) in self.config_dict.keys(): return self.config_dict[(d,Q)]['beta']
        else: return BETA_DEFAULT
        
    def get_beta_prune(self, d, Q):
        Q = float(Q)
        if (d, Q) in self.config_dict.keys(): return self.config_dict[(d,Q)]['beta_prune']
        else: return BETA_PRUNE_DEFAULT        
    
    def cuda(self):
        self.DEVICE = torch.device('cuda')
        
    def cpu(self):
        self.DEVICE = torch.device('cpu')
        
    def set_precision(self, prec: Literal['single', 'double']):
        assert(prec in ['single', 'double'])
        if prec == 'single':
            self.REAL_DTYPE = torch.float32
            self.COMPLEX_DTYPE = torch.complex64
        else:
            self.REAL_DTYPE = torch.float64
            self.COMPLEX_DTYPE = torch.complex128
            
cfg = Config()

# set default configs used in experiments

# Q = 1; d = 2^J
cfg.set_config(
    d=2, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.8,
    )
cfg.set_config(
    d=4, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.8,
    )
cfg.set_config(
    d=8, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.8,
    )
cfg.set_config(
    d=16, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.8,
    )

# Q = 1; d = 2^J*3
cfg.set_config(
    d=6, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.5,
    )
cfg.set_config(
    d=12, Q=1, 
    alpha=2,
    alpha_lin=2,
    alpha_start=1.5,
    )

# Q = 0.75
cfg.set_config(
    d=2, Q=0.75, 
    alpha=2.5,
    alpha_lin=2,
    alpha_start=1.2,
    beta=2.33,
    beta_prune=1.0
    )
cfg.set_config(
    d=4, Q=0.75, 
    alpha=3,
    alpha_lin=2,
    alpha_start=1.25,
    beta=2.5,
    beta_prune=1.0
    )
cfg.set_config(
    d=6, Q=0.75, 
    alpha=2.5,
    alpha_lin=2,
    alpha_start=1.5,
    beta=2.33,
    beta_prune=1.0
    )
cfg.set_config(
    d=8, Q=0.75, 
    alpha=3,
    alpha_lin=2,
    alpha_start=1.0,
    beta=2.5,
    beta_prune=1.0
    )