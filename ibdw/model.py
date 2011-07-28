# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import *
from generic_mbg import *
import generic_mbg
from ibdw import cut_matern, cut_gaussian
import scipy
from scipy import stats

__all__ = ['make_model']

# The parameterization of the cut between western and eastern hemispheres.
#
# t = np.linspace(0,1,501)
# 
# def latfun(t):
#     if t<.5:
#         return (t*4-1)*np.pi
#     else:
#         return ((1-t)*4-1)*np.pi
#         
# def lonfun(t):
#     if t<.25:
#         return -28*np.pi/180.
#     elif t < .5:
#         return -28*np.pi/180. + (t-.25)*3.5
#     else:
#         return -169*np.pi/180.
#     
# lat = np.array([latfun(tau)*180./np.pi for tau in t])    
# lon = np.array([lonfun(tau)*180./np.pi for tau in t])

# Tried: unconstrained scale, threshold = .001, .01
# Scale < 2, threshold = .001


constrained = True
# threshold_val = 1e-6
# max_p_above = 1e-6

threshold_val = 0.0001
max_p_above = 0.0001

def mean_fn(x,m):
    return pm.gp.zero_fn(x)+m

def make_model(lon,lat,input_data,covariate_keys,pos,neg):
    """
    This function is required by the generic MBG code.
    """
    
    # How many nuggeted field points to handle with each step method
    grainsize = 10

    # Unique data locations
    data_mesh, logp_mesh, fi, ui, ti = uniquify(lon, lat)
    
    s_hat = (pos+1.)/(pos+neg+2.)
        
    # The partial sill.
    amp = pm.Exponential('amp', .1, value=1.4)

    # The range parameters. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    scale = pm.Exponential('scale', .1, value=.07)
    @pm.potential
    def scale_constraint(scale=scale):
        if scale>.5:
            return -np.inf
        else:
            return 0

    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree', .01, 3, value=0.5, observed=True)

    # The nugget variance.
    V = pm.Exponential('V', .1, value=1)
    # @pm.potential
    # def V_constraint(V=V):
    #     if V<.1:
    #         return -np.inf
    #     else:
    #         return 0

    coef = np.array([-1.48556762,  0.28125179,  0.02261485,  0.02125477])

    def poly(x,coef=coef):
        return np.sum([c_*x**(power) for (power, c_) in enumerate(coef)], axis=0)

    def linkfn(x, a=[0,0], coef=coef):
        return pm.flib.stukel_invlogit(poly(x,coef), *a)

    def inverse_poly(y, coef=coef):
        poly = coef[::-1] + np.array([0,0,0,-y])
        roots = filter(lambda x: not x.imag, np.roots(poly))
        return np.array(roots).real

    def inverse_linkfn(y, a=[0,0], coef=coef, range=range):
        all_sol = inverse_poly(pm.flib.stukel_logit(y, *a), coef)
        # return all_sol[np.argmin(np.abs(all_sol))]
        if len(all_sol)>1:
            raise RuntimeError
        return all_sol[0]

    a0 = pm.Normal('a0',0,.1,value=0,observed=True)
    # a1 limits mixing.
    a1 = pm.Normal('a1',0,.1,value=0,observed=True)
    a = pm.Lambda('a',lambda a0=a0,a1=a1: [a0,a1])

    m = pm.Uninformative('m',value=-25)
    @pm.deterministic(trace=False)
    def M(m=m):
        return pm.gp.Mean(mean_fn, m=m)
    
    if constrained:
        @pm.potential
        def pripred_check(m=m,amp=amp,V=V,a=a):
            p_above = scipy.stats.distributions.norm.cdf(m-inverse_linkfn(threshold_val, a), 0, np.sqrt(amp**2+V))
            if p_above <= max_p_above:
                return 0.
            else:
                return -np.inf

    # Create the covariance & its evaluation at the data locations.
    facdict = dict([(k,1.e6) for k in covariate_keys])
    facdict['m'] = 0
    @pm.deterministic(trace=False)
    def C(amp=amp, scale=scale, diff_degree=diff_degree, ck=covariate_keys, id=input_data, ui=ui, facdict=facdict):
        """A covariance function created from the current parameter values."""
        eval_fn = CovarianceWithCovariates(pm.gp.matern.geo_rad, id, ck, ui, fac=facdict)
        return pm.gp.FullRankCovariance(eval_fn, amp=amp, scale=scale, diff_degree=diff_degree)

    sp_sub = pm.gp.GPSubmodel('sp_sub', M, C, logp_mesh, tally_f=False)
            
    # Make f start somewhere a bit sane
    sp_sub.f_eval.value = sp_sub.f_eval.value - np.mean(sp_sub.f_eval.value)

    # Loop over data clusters
    eps_p_f_d = []
    s_d = []
    data_d = []
    
    #coef = np.array([ -1.52718682e+00,   1.88227984e-01,  -2.03947095e-02,
    #     1.94652535e-02,   7.15159752e-04])
    

    for i in xrange(len(pos)/grainsize+1):
        sl = slice(i*grainsize,(i+1)*grainsize,None)        
        if len(pos[sl])>0:
            # Nuggeted field in this cluster
            eps_p_f_d.append(pm.Normal('eps_p_f_%i'%i, sp_sub.f_eval[fi[sl]], 1./V, value=pm.logit(s_hat[sl]), trace=False))            

            # Tomorrow: Empirically set the link function to the MLE?

            # The allele frequency
            s_d.append(pm.Lambda('s_%i'%i,lambda lt=eps_p_f_d[-1], a=a, coef=coef:  linkfn(lt, a, coef), trace=False))
            
            # The observed allele frequencies
            data_d.append(pm.Binomial('data_%i'%i, pos[sl]+neg[sl], s_d[-1], value=pos[sl], observed=True))
    
    # The field plus the nugget
    @pm.deterministic
    def eps_p_f(eps_p_fd = eps_p_f_d):
        """Concatenated version of eps_p_f, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(eps_p_fd)
            
    return locals()
