
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import brute
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
        self.fit_logL = -1*op.fun
        self.fit_BIC = self.BIC(
                len(self.fit_cannon_labels), 
                self.fit_logL)
        self.model_flux = np.interp(
            self.wav, 
            self.wav + self.wav * self.fit_cannon_labels[-1]/speed_of_light_kms, 
            fit_cannon_model(self.fit_cannon_labels[:-1]))
        self.model_residuals = self.flux - self.model_flux

    def fit_binary_fixed_components(self, primary_cannon_model, secondary_cannon_model):
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
                        for primary star (teff, logg, Fe/H, vsini, RV)
                secondary_cannon_model (tc.CannonModel): Cannon model component used 
                        for secondary star (teff, logg, vsini, RV)"""

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
        #print('initializing local search at Teff1={}, Teff2={}'.format(teff1_init, teff2_init))
        
        # initial labels + step size for local minimizer based on brute search outputs
        initial_labels = np.array([teff1_init, logg1_init, feh1_init, vsini1_init, 0, \
                  teff2_init, logg2_init, vsini2_init, 0])
        initial_steps = [50, 0.1, 0.01, 0.1, 0.5, 50, 0.1, 0.1, 0.5]
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

        # re-parameterize from log(vsini) to vsini
        op_minimize.binary_fit_cannon_labels = op_minimize.x.copy()
        op_minimize.binary_fit_cannon_labels[3] = 10**op_minimize.binary_fit_cannon_labels[3]
        op_minimize.binary_fit_cannon_labels[7] = 10**op_minimize.binary_fit_cannon_labels[7]
        
        # store binary model flux associated with maximum likelihood
        op_minimize.model_flux, _, _ = binary_model(
            op_minimize.binary_fit_cannon_labels[:5], 
            op_minimize.binary_fit_cannon_labels[[5,6,2,7,8]])
        
        return op_minimize

    def fit_binary(self):

        # run maximum likelihood optimization in hot + cool model regimes
        t0_fit = time.time()
        optimizers = {
            'hot1_cool2': self.fit_binary_fixed_components(
                self.hot_cannon_model, 
                self.cool_cannon_model),
            'cool1_cool2': self.fit_binary_fixed_components(
                self.cool_cannon_model, 
                self.cool_cannon_model),
            'hot1_hot2': self.fit_binary_fixed_components(
                self.hot_cannon_model, 
                self.hot_cannon_model)}
        #print('total time = {} seconds'.format(time.time()-t0_fit))
        
        # select best-fit binary from all regimes
        # (i.e., lowest -log(Likelihood))
        op = min(optimizers.values(), key=lambda x: x.fun)
        self.binary_fit_model = next(name for name, i in optimizers.items() if i is op)
        
        # re-parameterize from log(vsini) to vsini
        self.binary_fit_cannon_labels = op.binary_fit_cannon_labels.copy()

        # update spectrum attributes
        self.binary_fit_logL = -1*op.fun
        self.binary_fit_BIC = self.BIC(
            len(self.binary_fit_cannon_labels), 
            self.binary_fit_logL)
        self.binary_model_flux = op.model_flux
        self.binary_model_residuals = self.flux - self.binary_model_flux
        self.delta_BIC = self.fit_BIC-self.binary_fit_BIC

    def plot_binary(self):
        spec_tick_kwargs = {'axis':'x', 'length':8, 'direction':'inout'}
        fig = plt.figure(figsize=(15,10))
        fig = plt.figure(constrained_layout=True, figsize=(13,7))
        plt.rcParams['font.size']=13
        gs = fig.add_gridspec(3, 4, wspace = 0, hspace = 0)
        #gs.update(hspace=0)

        # spectrum + single star fit + binary fit
        ax1 = fig.add_subplot(gs[0:1, 0:3])
        ax1.errorbar(self.wav, self.flux+1, self.sigma, 
                color='k', ecolor='#E8E8E8', linewidth=1.75, elinewidth=4, zorder=0)
        ax1.plot(self.wav, self.model_flux+1, 'r-', alpha=0.8)
        ax1.plot(self.wav, self.binary_model_flux+1, '-', color='#4808c8', alpha=0.8)
        ax1.plot(self.wav, self.model_residuals, 'r-', alpha=0.8)
        ax1.plot(self.wav, self.binary_model_residuals, '-', color='#4808c8', alpha=0.8)
        ax1.set_xlim(self.wav[0], self.wav[-1])
        ax1.set_ylabel('normalized flux')
        ax1.tick_params(**spec_tick_kwargs)
        ax1.grid()

        # single star + binary residuals
        ax2 = fig.add_subplot(gs[1:2, 0:3])
        ax2.errorbar(self.wav, self.flux+1, self.sigma, 
                color='k', ecolor='#E8E8E8', linewidth=1.75, elinewidth=4, zorder=0)
        ax2.plot(self.wav, self.model_flux+1, 'r-', alpha=0.8)
        ax2.plot(self.wav, self.binary_model_flux+1, '-', color='#4808c8', alpha=0.8)
        ax2.plot(self.wav, self.model_residuals, 'r-', alpha=0.8)
        ax2.plot(self.wav, self.binary_model_residuals, '-', color='#4808c8', alpha=0.8)
        ax2.set_ylabel('normalized flux')
        ax2.set_xlim(5160,5190)
        ax2.tick_params(**spec_tick_kwargs)
        ax2.grid()

        # binary model components
        ax3 = fig.add_subplot(gs[2:, 0:3])
        ax3.errorbar(self.wav, self.flux+1, self.sigma, 
                color='k', ecolor='#E8E8E8', linewidth=1.75, elinewidth=4, zorder=0)
        ax3.plot(self.wav, self.model_flux+1, 'r-', alpha=0.8)
        ax3.plot(self.wav, self.binary_model_flux+1, '-', color='#4808c8', alpha=0.8)
        ax3.plot(self.wav, self.model_residuals, 'r-', alpha=0.8)
        ax3.plot(self.wav, self.binary_model_residuals, '-', color='#4808c8', alpha=0.8)
        ax3.set_xlim(5220,5240)
        ax3.set_xlabel('wavelength (nm)');ax3.set_ylabel('normalized flux')
        ax3.tick_params(labelbottom=True, **spec_tick_kwargs)
        ax3.grid()

        # 1D histogram: delta chisq
        ax4 = fig.add_subplot(gs[0:2, 3:])
        ax4.plot(self.cool_cannon_model.training_set_labels.T[0], 
            self.cool_cannon_model.training_set_labels.T[1], 
             'o', color='lightgrey')
        ax4.plot(self.hot_cannon_model.training_set_labels.T[0], 
            self.hot_cannon_model.training_set_labels.T[1], 
             'o', color='lightgrey')
        ax4.plot(self.binary_fit_cannon_labels[0], self.binary_fit_cannon_labels[1], 'o', color='#4808c8', mec='k')
        ax4.plot(self.binary_fit_cannon_labels[5], self.binary_fit_cannon_labels[6], 'o', color='#4808c8', mec='k')
        ax4.plot(self.fit_cannon_labels[0], self.fit_cannon_labels[1], 'r*', ms=15, mec='k')
        ax4.set_xlim(7000,2800);ax4.set_ylim(5.4,3.4)
        ax4.set_yticks(np.arange(3.5,6,0.5))
        ax4.set_xlabel('Teff (K)');ax4.set_ylabel('logg (dex)')


