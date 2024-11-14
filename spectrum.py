
from scipy.optimize import brute
from scipy.optimize import minimize
from spectrum_utils import *
import numpy as np
import time

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

    def op_bounds(self, cannon_model):
        """Determine bounds of scipy.opimize.minimize
        based on the minimum and maximum training set labels
        for a given Cannon model."""

        # bounds from training sets
        min_bounds = np.min(cannon_model.training_set_labels, axis=0)
        max_bounds = np.max(cannon_model.training_set_labels, axis=0)

        # add in RV shift and bound to (-10, 10) km/s
        min_bounds = np.append(min_bounds, -10)
        max_bounds = np.append(max_bounds, 10)

        # re-parameterize from vsini to log(vsini)
        # note: log(vsini)=-2-1 is vsini=0.01-10km/s
        min_bounds[-2] = -2
        max_bounds[-2] = 1

        return tuple(zip(min_bounds, max_bounds))
    
    
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

        def negative_logL(param, cannon_model):
            """Negative log-likelihood associated with set of Cannon model parameters.
            We use the Bayesian Likelihood formula for this calculation.
            """
            # re-parameterize from log(vsini) to vsini
            param[-2] = 10**param[-2]

            # determine model, error term based on piecewise model
            sn2 = self.sigma**2 + cannon_model.s2

            # evaluate cannon model at labels of interest
            model = cannon_model(param[:-1])
            
            # apply RV shift
            delta_w = self.wav * param[-1]/speed_of_light_kms
            model_shifted = np.interp(self.wav, self.wav + delta_w, model)
        
            # compute log-likelihood
            term_in_brackets = (self.flux - model_shifted)**2/sn2 + np.log(2*np.pi*sn2)
            negative_logLikelihood = (1/2)*np.sum(term_in_brackets)

            return negative_logLikelihood

        # determine initial labels
        cool_param_init = self.cool_cannon_model._fiducials.copy()[2:].tolist() + [0]
        hot_param_init = self.hot_cannon_model._fiducials.copy()[2:].tolist() + [0]

        # re-parameterize from vsini to log(vsini)
        cool_param_init[-2] = np.log10(cool_param_init[-2])
        hot_param_init[-2] = np.log10(hot_param_init[-2])
        
        # coarse brute search to determine initial Teff, logg
        teff_hr = [3000, 3500, 4000, 4500, 5000, 5500, 5750, 6000, 6500, 5250]
        logg_hr = [5, 4.8, 4.7, 4.6, 4.5, 4.45, 4, 4.35, 4.15, 3.8]
        hr_init = [[teff, logg] for teff, logg in zip(teff_hr, logg_hr)]
        negative_logL_hr_cool = [
            negative_logL(i + cool_param_init, self.cool_cannon_model) for i in hr_init[:5]]
        negative_logL_hr_hot = [
            negative_logL(i + hot_param_init, self.hot_cannon_model) for i in hr_init[5:]]

        # update initial conditions with brute search outputs
        cool_param_init = hr_init[:5][np.argmin(negative_logL_hr_cool)] + cool_param_init
        hot_param_init = hr_init[5:][np.argmin(negative_logL_hr_hot)] + hot_param_init

        
        # TO DO: add RV bounds to optimizer
        # then make sure binary bounds reflect this
        # then git commit
        
        # fit spectrum with hot + cool cannon models
        op_cool = minimize(
            negative_logL, 
            cool_param_init, 
            args=(self.cool_cannon_model), 
            bounds = self.op_bounds(self.cool_cannon_model),
            method = 'Nelder-Mead')
        
        op_hot = minimize(
            negative_logL, 
            hot_param_init, 
            args=(self.hot_cannon_model), 
            bounds = self.op_bounds(self.hot_cannon_model),
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
        self.fit_cannon_labels[-2] = 10**self.fit_cannon_labels[-2]
        
        # update spectrum attributes
        self.fit_logLikelihood = -1*op.fun
        self.fit_BIC = self.BIC(
                len(self.fit_cannon_labels), 
                self.fit_logLikelihood)
        self.model_flux = np.interp(
            self.wav, 
            self.wav + self.wav * self.fit_cannon_labels[-1]/speed_of_light_kms, 
            fit_cannon_model(self.fit_cannon_labels[:-1]))
        self.model_residuals = self.flux - self.model_flux

    def maxL_binary(self, primary_cannon_model, secondary_cannon_model):
        # initial conditions based on component cannon models
        _, logg1_init, feh1_init, vsini1_init = primary_cannon_model._fiducials.copy()
        _, logg2_init, _, vsini2_init = secondary_cannon_model._fiducials.copy()
        
        # re-parameterize from vsini to log(vsini)
        vsini1_init, vsini2_init = np.log10(vsini1_init), np.log10(vsini2_init)
        
        # optimizer bounds set by cannon training set
        # with RV offset=-10-10km/s
        primary_op_bounds = self.op_bounds(primary_cannon_model)
        secondary_op_bounds = self.op_bounds(secondary_cannon_model)
        binary_op_bounds = primary_op_bounds + \
                            (secondary_op_bounds[0],) + (secondary_op_bounds[1],) + \
                            (secondary_op_bounds[3],) + (secondary_op_bounds[4],)
        
        def binary_model(param1, param2):
            """Calculate binary model associated with set of parameters
            a particular set of model parameters. Also returns s2 associated with 
            primary and secondary components (weighted based on relative flux)
            
            Args:
                primary_cannon_model (tc.CannonModel): Cannon model component used 
                        for primary star
                secondary_cannon_model (tc.CannonModel): Cannon model component used 
                        for secondary star"""

            # compute relative flux based on temperature
            W1, W2 = flux_weights(param1[0], param2[0], self.wav)

            # compute single star models for both components
            flux1  = primary_cannon_model(param1[:-1])
            flux2 = secondary_cannon_model(param2[:-1])
            
            # compute intrinsic model errors (s2)
            s2_1 = W1*primary_cannon_model.s2
            s2_2 = W2*secondary_cannon_model.s2

            # shift flux1, flux2 according to drv
            delta_w1 = self.wav * param1[-1]/speed_of_light_kms
            delta_w2 = self.wav * param2[-1]/speed_of_light_kms
            flux1_shifted = np.interp(self.wav, self.wav + delta_w1, flux1)
            flux2_shifted = np.interp(self.wav, self.wav + delta_w2, flux2)

            # compute weighted sum of primary, secondary
            model = W1*flux1_shifted + W2*flux2_shifted

            return model, s2_1, s2_2
        
        def negative_logL(params):
            """Negative log-likelihood associated with set of composite Cannon model parameters.
            We use the Bayesian Likelihood formula for this calculation.
            """ 
            # re-parameterize from log(vsini) to vsini
            params[3] = 10**params[3]
            params[7] = 10**params[7]
            
            # store primary, secondary parameters
            param1 = params[:5]
            param2 = params[[5,6,2,7,8]]

            # prevent model from Teff outside Pecaut&Mamajek interpolation range
            # and require primary Teff> secondary Teff
            if 2450>param1[0] or 34000<param1[0]:
                return np.inf
            elif 2450>param2[0] or 34000<param2[0]:
                return np.inf
            elif param2[0]>param1[0]:
                return np.inf

            # determine model, error term based on piecewise model
            model, s2_primary, s2_secondary = binary_model(param1, param2)
            sn2 = self.sigma**2 + s2_primary + s2_secondary

            # compute log-likelihood
            term_in_brackets = (self.flux - model)**2/sn2 + np.log(2*np.pi*sn2)
            negative_logLikelihood = (1/2)*np.sum(term_in_brackets)

            return negative_logLikelihood
            
        def negative_logL_wrapper(teff_params):
            """Wrapper function that computes residuals while only varying teff1, teff_ratio,
            used only for coarse brute search. Other labels are fixed to values found
            with single star optimizer."""

            params = np.array([teff_params[0], logg1_init, feh1_init, vsini1_init, 0, \
                      teff_params[1], logg2_init, vsini2_init, 0])
            negative_logLikelihood = negative_logL(params)

            return negative_logLikelihood   
        
        # perform coarse brute search for ballpark teff1, teff2
        #t0_brute = time.time()
        teff_ranges = (
            slice(primary_op_bounds[0][0], primary_op_bounds[0][1], 100), # teff1
            slice(secondary_op_bounds[0][0], secondary_op_bounds[0][1], 100)) # teff2
        op_brute = brute(negative_logL_wrapper, teff_ranges, finish=None)
        teff1_init, teff2_init = op_brute
        #print('total time for brute search: {} seconds'.format(time.time()-t0_brute))
        print('initializing brute search at Teff1={}, Teff2={}'.format(teff1_init, teff2_init))
        
        # initial labels + step size for local minimizer based on brute search outputs
        initial_labels = np.array([teff1_init, logg1_init, feh1_init, vsini1_init, 0, \
                  teff2_init, logg2_init, vsini2_init, 0])
        initial_steps = [10, 0.1, 0.01, 0.1, 0.5, 10, 0.1, 0.1, 0.5]
        initial_simplex = [initial_labels] + [np.array(initial_labels) + \
                                              np.eye(len(initial_labels))[i] * initial_steps[i]\
                                              for i in range(len(initial_labels))]

        # perform localized search at minimum from brute search
        #t0_minimize = time.time()
        op_minimize = minimize(
            negative_logL, 
            x0=initial_labels,
            bounds = binary_op_bounds,
            method='Nelder-Mead',
            options={'initial_simplex':initial_simplex,'fatol':1,'xatol':1})
        #print('final Teff1={}, Teff2={}'.format(op_minimize.x[0], op_minimize.x[5]))
        #print('total time for local optimizer: {} seconds'.format(time.time()-t0_minimize))
        
        # store binary model flux associated with maximum likelihood
        op_minimize.model_flux, _, _ = binary_model(
            op_minimize.x[:5], 
            op_minimize.x[[5,6,2,7,8]])
        
        return op_minimize

    def fit_binary(self):
        # run maximum likelihood optimization in hot + cool model regimes
        t0_fit = time.time()
        op_hot1_cool2 = self.maxL_binary(self.hot_cannon_model, self.cool_cannon_model)
        op_cool1_cool2 = self.maxL_binary(self.cool_cannon_model, self.cool_cannon_model)
        op_hot1_hot2 = self.maxL_binary(self.hot_cannon_model, self.hot_cannon_model)

        # TEMPORARY: store different optimizer model fluxes
        self.hot1_cool2_flux = op_hot1_cool2.model_flux
        self.hot1_hot2_flux = op_hot1_hot2.model_flux
        self.cool1_cool2_flux = op_cool1_cool2.model_flux

        print('total time = {} seconds'.format(time.time()-t0_fit))
        
        # select best-fit binary from all regimes
        # (i.e., lowest -log(Likelihood))
        op = min([op_hot1_cool2, op_cool1_cool2, op_hot1_hot2], key=lambda x: x.fun)
        
        # re-parameterize from log(vsini) to vsini
        self.binary_fit_cannon_labels = op.x
        self.binary_fit_cannon_labels[3] = 10**self.binary_fit_cannon_labels[3]
        self.binary_fit_cannon_labels[7] = 10**self.binary_fit_cannon_labels[7]

        # update spectrum attributes
        self.binary_fit_logLikelihood = -1*op.fun
        self.binary_fit_BIC = self.BIC(
            len(self.binary_fit_cannon_labels), 
            self.binary_fit_logLikelihood)
        self.binary_model_flux = op.model_flux
        self.binary_model_residuals = self.flux - self.binary_model_flux
        self.delta_BIC = self.fit_BIC-self.binary_fit_BIC



