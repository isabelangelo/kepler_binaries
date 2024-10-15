import numpy as np
from scipy.optimize import leastsq
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

	def piecewise_cannon_model(self, param, return_flux = False):
		"""Piecewise Cannon model that combines hot star component 
		and cool star component"""

		# determine which component model to call
		if param[0]<5250:
		    cannon_model = self.cool_cannon_model
		else:
		    cannon_model = self.hot_cannon_model

		# return model or model flux
		if return_flux:
			return cannon_model(param)
		else:
			return cannon_model		

	def fit_single_star(self):
		""" Run the test step on the ra (similar to the Cannon 2 
		test step, but we mask the sodium + telluric lines)"""
		def residuals(param):
			# determine piecewise component to use
			cannon_model = self.piecewise_cannon_model(param)
			# compute residuals
			err2 = self.sigma**2 + cannon_model.s2
			weights = 1/np.sqrt(err2)
			model = cannon_model(param)
			resid = weights * (model - self.flux)
			return resid

		def residuals_wrapper(teff_logg_param):
			# Fe/H fixed to 0, vsini fixed to 3 in brute search
			param = teff_logg_param + [0,3]
			return sum(residuals(param)**2)

		# brute search through 4 points on HR diagram to determine initial labels
		hr_initial_params = ([3500,4.75],[4500,4.65],[5500,4.5],[6500,4.2])
		op_brute = [(p, residuals_wrapper(p)) for p in hr_initial_params]
		optimal_point, min_value = min(op_brute, key=lambda x: x[1])

		# local optimizer initialized at brute search results
		initial_labels = optimal_point + [0,3]
		op_local = leastsq(residuals,x0=initial_labels, full_output=True)

		# update model attributes
		self.fit_cannon_labels = op_local[0]
		self.fit_chisq = np.sum(op_local[2]["fvec"]**2)
		self.model_flux = self.piecewise_cannon_model(self.fit_cannon_labels, return_flux=True)



