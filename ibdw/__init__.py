# from mcmc import *
from generic_mbg import stukel_invlogit, fast_inplace_mul, fast_inplace_square, fast_inplace_scalar_add
import pymc as pm
from cut_geographic import cut_geographic, hemisphere
import ibdw
import numpy as np
import os
root = os.path.split(ibdw.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

import cg
from cg import *

cut_matern = pm.gp.cov_utils.covariance_wrapper_with_diag('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
cut_gaussian = pm.gp.cov_utils.covariance_wrapper_with_diag('gaussian', 'pymc.gp.cov_funs.isotropic_cov_funs', {}, 'cut_geographic', 'cg')

nugget_labels = {'sp_sub': 'V'}
obs_labels= {'sp_sub': 'eps_p_f'}
non_cov_columns = {'pos': 'float', 'neg': 'float'}

def check_data(input):
    if np.any(input.pos+input.neg)==0:
        raise ValueError, 'Some sample sizes are zero.'
    if np.any(np.isnan(input.pos)) or np.any(np.isnan(input.neg)):
        raise ValueError, 'Some NaNs in input'
    if np.any(input.pos<0) or np.any(input.neg<0):
        raise ValueError, 'Some negative values in pos and neg'
        
def allele(sp_sub, a):
    allele = sp_sub.copy('F')
    allele = stukel_invlogit(allele, *a)
    return allele

def hw_homo(sp_sub, a):
    hom = allele(sp_sub, a)
    fast_inplace_mul(hom,hom)
    return hom
    
def hw_hetero(sp_sub, a):
    p = allele(sp_sub, a)
    q = fast_inplace_scalar_add(-p,1)
    fast_inplace_mul(p,q)
    return 2*p
    
def hw_any(sp_sub, a):
    homo = hw_homo(sp_sub, a)
    hetero = hw_hetero(sp_sub, a)
    return hetero+homo

# map_postproc = [allele, hw_hetero, hw_homo, hw_any]
# map_postproc = [allele hw_homo, hw_any]
map_postproc = [allele, hw_homo, hw_hetero, hw_any]

def validate_allele(data):
    obs = data.pos
    n = data.pos + data.neg
    def f(sp_sub, a, n=n):
        return pm.rbinomial(n=n,p=pm.stukel_invlogit(sp_sub, *a))
    return obs, n, f

validate_postproc = [validate_allele]

regionlist=['Free','Epidemic','Hypoendemic','Mesoendemic','Hyperendemic','Holoendemic']
    
def area_allele(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, a, x):
        p = pm.stukel_invlogit(sp_sub(x), *a)
        return p

    g = {gc.keys()[0]: f}
    
    return h,g

def area_hw_hetero(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, a, x):
        p = pm.stukel_invlogit(sp_sub(x), *a)
        return 2*p*(1-p)

    g = {gc.keys()[0]: f}
    
    return h,g

def area_hw_homo(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, a, x):
        p = pm.stukel_invlogit(sp_sub(x), *a)
        return p**2

    g = {gc.keys()[0]: f}
    
    return h,g

def area_hw_any(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, a, x):
        p = pm.stukel_invlogit(sp_sub(x), *a)
        return 2*p*(1-p)+p**2

    g = {gc.keys()[0]: f}
    
    return h,g

areal_postproc = [area_allele, area_hw_homo, area_hw_hetero, area_hw_any]

def mcmc_init(M):
    scalar_vars = [M.amp, M.amp_short_frac, M.scale_short, M.scale_long, M.diff_degree, M.a, M.m]
    scales = dict([(k,.001) for k in scalar_vars])
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_vars, scales)
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f)
                    
metadata_keys = ['fi','ti','ui']

from model import *