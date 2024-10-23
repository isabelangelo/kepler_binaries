import numpy as np
from scipy.optimize import leastsq
from spectrum_utils import *

import numpy as np
from scipy.optimize import minimize
from spectrum_utils import *

class Spectrum(object):
    """
    HIRES Spectrum object

    Args:
        flux (np.array): flux of object, post-wavelet transform
        sigma (np.array): flux errors of object
        order_numbers (int or array-like): order numbers to include, 1-16 for HIRES r chip
                        e.g., 4, [1,2,6,15,16]
        cool_cannon_model (tc.CannonModel): cool star Cannon model object to use to 
                        model spectrum, cool star component for piecewise model
        hot_cannon_model (tc.CannonModel): hot star Cannon model object to use to 
                        model spectrum, hot star component for piecewise model

    """
    def __init__(self, flux, sigma, order_numbers, cool_cannon_model, hot_cannon_model):

        # store spectrum information, mask telluric features
        self.flux = np.array(flux)
        self.sigma = np.array(sigma)

        # convert single order to list
        if type(order_numbers)==int:
            self.order_numbers = [order_numbers]
        else:
            self.order_numbers = order_numbers

        # store order wavelength
        self.wav = wav_data[[i-1 for i in self.order_numbers]].flatten()

        # compute telluric mask
        self.mask = np.empty_like(self.flux).astype(bool)
        self.mask.fill(True)
        for n, row in mask_table_cut.iterrows():
            start = row['minw']
            end = row['maxw']
            self.mask[(self.wav>start) & (self.wav<end)] = False

        # store cannon model information
        self.cool_cannon_model = cool_cannon_model
        self.hot_cannon_model = hot_cannon_model
    
    def BIC(self, k, logLikelihood):
        """Compute Bayesian Information Criterion for Cannon model
        Args:
            k (int): number of model free parameters
            logL: log of Bayesian Likelihood"""
        term1 = k*np.log(len(self.flux))
        term2 = 2*logLikelihood
        return term1 - term2

    def fit_single_star(self):
        """ Run the test step on the spectrum (similar to the Cannon 2 
        test step, but we mask telluric lines and compute the best-fit
        based on the minimum -log(Likelihood) )
        """
        def op_bounds(cannon_model):
            """Determine bounds of scipy.opimize.minimize
            based on the minimum and maximum training set labels
            for a given Cannon model."""
            
            min_bounds = np.min(cannon_model.training_set_labels, axis=0)
            max_bounds = np.max(cannon_model.training_set_labels, axis=0)
            
            return tuple(zip(min_bounds, max_bounds))
        
        def negative_logL(param, cannon_model):
            """Negative log-likelihood associated with set of Cannon model parameters.
            We use the Bayesian Likelihood formula for this calculation.
            """
            # re-parameterize from log(vsini) to vsini
            param[-1] = 10**param[-1]

            # determine model, error term based on piecewise model
            sn2 = self.sigma**2 + cannon_model.s2
            
            # compute log-likelihood
            model = cannon_model(param)
            term_in_brackets = (self.flux - model)**2/sn2 + np.log(2*np.pi*sn2)
            negative_logLikelihood = (1/2)*np.sum(term_in_brackets)
            
            #print(param, negative_logLikelihood)
            
            return negative_logLikelihood

        # determine initial labels
        cool_param_init = self.cool_cannon_model._fiducials.copy()
        hot_param_init = self.hot_cannon_model._fiducials.copy()
        
        # re-parameterize from vsini to log(vsini)
        cool_param_init[-1] = np.log10(cool_param_init[-1])
        hot_param_init[-1] = np.log10(hot_param_init[-1])
        
        # fit spectrum with hot + cool cannon models
        op_cool = minimize(
            negative_logL, 
            cool_param_init, 
            args=(self.cool_cannon_model), 
            bounds = op_bounds(self.cool_cannon_model),
            method = 'Nelder-Mead')
        #import pdb;pdb.set_trace()
        op_hot = minimize(
            negative_logL, 
            hot_param_init, 
            args=(self.hot_cannon_model), 
            bounds = op_bounds(self.hot_cannon_model),
            method = 'Nelder-Mead')
        
        # select best-fit between hot + cool models
        if op_cool.fun<op_hot.fun:
            op = op_cool
            fit_cannon_model = self.cool_cannon_model
            self.fit_model = 'cool'
        else:
            op = op_hot
            fit_cannon_model = self.hot_cannon_model
            self.fit_model = 'hot'
        
        # re-parameterize from log(vsini) to vsini
        self.fit_cannon_labels = op.x
        self.fit_cannon_labels[-1] = 10**self.fit_cannon_labels[-1]
        
        # update spectrum attributes
        self.fit_logLikelihood = -1*op.fun
        self.fit_BIC = self.BIC(
            len(self.fit_cannon_labels), 
            self.fit_logLikelihood)
        self.model_flux = fit_cannon_model(self.fit_cannon_labels)
        self.model_residuals = self.flux - self.model_flux



