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

    def piecewise_cannon_model(self, param):
        """Piecewise Cannon model that combines hot star component 
        and cool star component"""

        # determine which component model to call
        if param[0]<5250:
            cannon_model = self.cool_cannon_model
        else:
            cannon_model = self.hot_cannon_model
        # return model flux + associated sw
        return cannon_model(param), cannon_model.s2
    
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
        based on the maximum Likelihood)"""
        
        def negative_logL(param):
            """Negative log-likelihood associated with set of Cannon model parameters.
            We use the Bayesian Likelihood formula for this calculation.
            """
            # re-parameterize from log(vsini) to vsini
            param[-1] = 10**param[-1]

            # determine model, error term based on piecewise model
            model, s2 = self.piecewise_cannon_model(param)
            sn2 = self.sigma**2 + s2
            
            # compute log-likelihood
            term_in_brackets = (self.flux - model)**2/sn2 + np.log(2*np.pi*sn2)
            negative_logLikelihood = (1/2)*np.sum(term_in_brackets)
            
            print(param, negative_logLikelihood)
            
            return negative_logLikelihood

        # brute search through 4 points on HR diagram to determine initial labels
        # Fe/H fixed to 0dex, vsini fixed to 3km/s
        feh_init, logvsini_init = 0, np.log10(3)
        hr_init_params = ([3500,4.75],[4500,4.65],[5500,4.5],[6500,4.2])
        op_brute = [(p, negative_logL(p+[feh_init, logvsini_init])) for p in hr_init_params]
        optimal_point, min_value = min(op_brute, key=lambda x: x[1])
        print('initial conditions from brute search:', optimal_point)

        # local optimizer initialized at brute search results
        initial_labels = optimal_point + [feh_init, logvsini_init]
        op_local = minimize(negative_logL, x0=initial_labels, method='L-BFGS-B')

        # store best-fit cannon labels, 
        # re-parameterize from log(vsini) to vsini
        self.fit_cannon_labels = op_local.x
        self.fit_cannon_labels[-1] = 10**self.fit_cannon_labels[-1]
        
        # update spectrum attributes
        self.fit_logLikelihood = -1*op_local.fun
        self.fit_BIC = self.BIC(
            len(self.fit_cannon_labels), 
            self.fit_logLikelihood)
        self.model_flux, _ = self.piecewise_cannon_model(self.fit_cannon_labels)
        self.model_residuals = self.flux - self.model_flux



