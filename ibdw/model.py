# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import basic_spatial_submodel

__all__ = ['make_model']
    
def make_model(s_obs,a_obs,lon,lat,from_ind,covariate_values):
    """
    """

    V = pm.Exponential('V',.1,value=1.)

    init_OK = False
    while not init_OK:
        try:        
            # Space-time component
            sp_sub = basic_spatial_submodel(lon, lat, covariate_values)        

            # The field evaluated at the uniquified data locations
            f = pm.MvNormalCov('f', sp_sub['M_eval'], sp_sub['C_eval'])            
        
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
        
    # The field plus the nugget
    eps_p_f = pm.Normal('eps_p_f', f[from_ind], V)
    
    s = pm.InvLogit('s',eps_p_f)

    data = pm.Binomial('data', s_obs + a_obs, s, value=s_obs, observed=True)

    out = locals()
    out.pop('sp_sub')
    out.update(sp_sub)

    return out