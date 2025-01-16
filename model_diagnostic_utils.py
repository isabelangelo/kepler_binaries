"""
Defines functions that are used to generate diagnostic plots
for Cannon and Payne models
"""
from specmatchemp.spectrum import read_hires_fits
from astropy.io import fits
import The_Payne.fitting
import matplotlib.pyplot as plt
import thecannon as tc
import pandas as pd
import numpy as np
import spectrum

# spectra of binary + single star samples
# NOTE: for these purposes I am using the single stars with degraded errors
single_flux = pd.read_csv('./data/spectrum_dataframes/raghavan_single_flux_dwt_degraded.csv')
single_sigma = pd.read_csv('./data/spectrum_dataframes/raghavan_single_sigma_dwt_degraded.csv')
binary_flux = pd.read_csv('./data/spectrum_dataframes/kraus_binary_flux_dwt.csv')
binary_sigma = pd.read_csv('./data/spectrum_dataframes/kraus_binary_sigma_dwt.csv')

# labels of binary + single star samples
single_labels = pd.read_csv('./data/label_and_metric_dataframes/raghavan_single_labels.csv')
binary_labels = pd.read_csv('./data/label_and_metric_dataframes/kraus_binary_labels.csv')

def read_payne_model(payne_model_path):
	"""loads in a neural net from file output from the Payne training"""
	tmp = np.load(payne_model_path + 'NN_normalized_spectra.npz')
	w_array_0 = tmp["w_array_0"]
	w_array_1 = tmp["w_array_1"]
	w_array_2 = tmp["w_array_2"]
	b_array_0 = tmp["b_array_0"]
	b_array_1 = tmp["b_array_1"]
	b_array_2 = tmp["b_array_2"]
	x_min = tmp["x_min"]
	x_max = tmp["x_max"]
	tmp.close()
	NN_coeffs = (w_array_0, w_array_1, w_array_2, 
	             b_array_0, b_array_1, b_array_2, 
	             x_min, x_max)
	return NN_coeffs

def fit_metrics(flux, sigma, model_type, order_numbers, cool_model=None, hot_model=None):
	"""
	Determines the fit statistics (sum of squared residuals, chi-squared)
	of a given spectrum using either a Cannon or Payne spectral model

	Args:
	    flux (np.array or pandas.core.series.Series): spectrum flux
	    sigma (np.array or pandas.core.series.Series): spectrum flux errors
	    model_type (str): 'cannon', 'piecewise_cannon' or 'payne'
	    order_numbers: HIRES orders included in trained model
	    cool_model (tc.CannonModel or None):
	        - if model_type='cannon' or 'payne', this should be None
	        - if model_type='piecewise_cannon', this is the cool star 
	            component fo the piecewise Cannon model, output from 
	            tc.CannonModel.read()
	    hot_model (tc.CannonModel or tuple):
	        - if model_type='cannon', this is the Cannon model output from
	            tc.CannonModel.read()
	        - if model_type='payne', this is the tuple output of read_payne_model()
	        - if model_type='piecewise_cannon', this is the hot star 
	            component fo the piecewise Cannon model, output from 
	            tc.CannonModel.read()      
	Returns:
	    sum_resid_sq (float): sum(data-model)^2 associated with best-fit model
	    chi_sq (float): sum(data-model)^2/sigma^2 associated with best-fit model
	    fit_labels (np.array): labels associated with best-fit model
	    
	"""

	if model_type=='cannon' or model_type=='piecewise_cannon':
	    
	    # determine best-fit single star
	    spec = spectrum.Spectrum(flux, sigma, order_numbers, cool_model, hot_model)
	    spec.fit_single_star()
	    data_minus_model = flux - spec.model_flux
	    fit_labels = spec.fit_cannon_labels
	    
	elif model_type=='payne':
	    
	    # model-specific wavelength, mask data
	    wavelength = spectrum.wav_data[[i-1 for i in order_numbers]].flatten()
	    mask = np.full(len(wavelength), False) # not masking any pixels currently
	    
	    # determine residuals of best-fit single star
	    popt, pstd, model_flux = The_Payne.fitting.fit_normalized_spectrum_single_star_model(
	        norm_spec = flux, spec_err = sigma,
	        NN_coeffs = hot_model, 
	        wavelength = wavelength, mask=mask, p0 = None)
	    data_minus_model = flux - model_flux
	    fit_labels = popt

	# fit statistics
	sum_resid_sq = sum(data_minus_model**2)
	chi_sq = sum(data_minus_model**2/sigma**2)

	return sum_resid_sq, chi_sq, fit_labels

def sample_metrics(sample, model_type, order_numbers, cool_model=None, hot_model=None):
	"""
	Stores the goodness-of-fit metrics and labels associated with the
	single star validation sample from Raghavan et al. (2010)
	for a given Cannon or Payne model.

	Args:
	    sample (str): 'single' or 'binary', determines validation which sample 
	        to compute metrics for (raghavan single star or kraus binary sample)
	    model_type (str): 'cannon', 'piecewise_cannon' or 'payne'
	    order_numbers: HIRES orders included in trained model
	    cool_model (tc.CannonModel or None):
	        - if model_type='cannon' or 'payne', this should be None
	        - if model_type='piecewise_cannon', this is the cool star 
	            component fo the piecewise Cannon model, output from 
	            tc.CannonModel.read()
	    hot_model (tc.CannonModel or tuple):
	        - if model_type='cannon', this is the Cannon model output from
	            tc.CannonModel.read()
	        - if model_type='payne', this is the tuple output of read_payne_model()
	        - if model_type='piecewise_cannon', this is the hot star 
	            component fo the piecewise Cannon model, output from 
	            tc.CannonModel.read()  
	Returns:
	    metric_df: Dataframe with labels + fit metrics for single star sample
	    computed with model of interest
	"""

	# determine flux, sigma and labels for given sample
	if sample=='single':
	    flux_df = single_flux.query('order_number in @order_numbers')
	    sigma_df = single_sigma.query('order_number in @order_numbers')
	    labels = single_labels
	    
	elif sample=='binary':
	    flux_df = binary_flux.query('order_number in @order_numbers')
	    sigma_df = binary_sigma.query('order_number in @order_numbers')
	    labels = binary_labels

	keys = ['id_starname', 'sum_resid_sq', 'chi_sq', 'model_teff', 'model_logg',\
	       'model_feh', 'model_vsini', 'model_psf', 'model_rv']
	metric_data = []

	for id_starname in labels.id_starname.to_numpy():
	    
	    # specrtrum data
	    flux = flux_df[id_starname]
	    sigma = sigma_df[id_starname]

	    # model goodness-of-fit metrics
	    sum_resid_sq, chi_sq, fit_labels = fit_metrics(flux, sigma, 
	                  model_type, order_numbers, 
	                  cool_model=cool_model, hot_model=hot_model)

	    # if model doesn't include psf, set to nan in dataframe
	    if len(fit_labels)<6:
	        fit_labels = np.insert(fit_labels, -1, np.nan, axis=None)

	    # combine data for dataframe
	    values = [id_starname, sum_resid_sq, chi_sq] + [i for i in fit_labels]
	    metric_data.append(dict(zip(keys,values)))

	metric_df = pd.DataFrame(metric_data)
	metric_df = pd.merge(labels, metric_df)

	return metric_df

def plot_diagnostics(single_metrics, binary_metrics, title_str):
	"""
	Create + save diagnostic plots from metrics associated with model of interest

	Args:
	    single_metrics (pd.DataFrame): labels + metrics for single star sample 
	        computed with model of interest
	    binary_metrics (pd.DataFrame): labels + metrics for binary sample 
	        computed with model of interest
	    model_name (str): descriptive title for plot to distinguish model of 
	        interest from other trained models (include order number, with/without PSF,
	        whether or not cool stars are included, etc.)
	"""

	plt.figure(figsize=(15,20))
	plt.rcParams['font.size']=15

	# histograms of model goodness-of-fit metrics
	plt.subplot(321)
	sum_resid_sq_bins = np.histogram(np.hstack((single_metrics.sum_resid_sq,binary_metrics.sum_resid_sq)), bins=50)[1]
	plt.hist(single_metrics.sum_resid_sq, bins=sum_resid_sq_bins, label='single sample', 
	     histtype='step', color='k', lw=2, alpha=1)
	plt.hist(binary_metrics.sum_resid_sq, bins=sum_resid_sq_bins, label='binary sample', 
	     histtype='step', color='cornflowerblue', lw=2, alpha=0.7)
	plt.xlabel(r'$\Sigma({\rm data - model})^2$');plt.ylabel('number of stars')
	plt.legend(loc='upper right')

	plt.subplot(322)
	chi_sq_bins = np.histogram(np.hstack((single_metrics.chi_sq,binary_metrics.chi_sq)), bins=50)[1]
	plt.hist(single_metrics.chi_sq, bins=chi_sq_bins, label='single sample', 
	     histtype='step', color='k', lw=2, alpha=1)
	plt.hist(binary_metrics.chi_sq, bins=chi_sq_bins, label='binary sample', 
	     histtype='step', color='cornflowerblue', lw=2, alpha=0.7)
	plt.xlabel(r'$\chi^2 = ({\rm data - model})^2/\sigma^2$');plt.ylabel('number of stars')
	plt.legend(loc='upper right')

	# binary sample separation vs delta_magnitude
	kraus2016_binary_metrics = binary_metrics.query('source=="Kraus 2016"')
	plt.subplot(323)
	plt.scatter(binary_metrics.sep_mas, binary_metrics.dmag, 
	        c=binary_metrics.sum_resid_sq, cmap='viridis_r',
	       vmin=sum_resid_sq_bins[0],
	       vmax=sum_resid_sq_bins[10], edgecolors='dimgrey', s=105)

	plt.xscale('log');plt.colorbar(location='top', label=r'$\Sigma({\rm data - model})^2$')
	plt.xticks([10,100,1000]);plt.ylim(6,-0.2);plt.xlim(8,1000)
	plt.xlabel('binary separation (mas)');plt.ylabel(r'$\Delta{\rm mag}$')

	plt.subplot(324)
	plt.scatter(binary_metrics.sep_mas, binary_metrics.dmag, 
	        c=binary_metrics.chi_sq, cmap='viridis_r',
	       vmin=chi_sq_bins[0],
	       vmax=chi_sq_bins[5], edgecolors='dimgrey', s=105)
	plt.xscale('log');plt.colorbar(location='top', label=r'$\chi^2 = ({\rm data - model})^2/\sigma^2$')
	plt.xticks([10,100,1000]);plt.ylim(6,-0.2);plt.xlim(8,1000)
	plt.xlabel('binary separation (mas)');plt.ylabel(r'$\Delta{\rm mag}$')

	# binary sample model Teff vs delta_magnitude
	plt.subplot(325)
	plt.scatter(binary_metrics.cks_teff, binary_metrics.dmag, 
	        c=binary_metrics.sum_resid_sq, cmap='inferno_r',
	       vmin=sum_resid_sq_bins[0],
	       vmax=sum_resid_sq_bins[10], edgecolors='dimgrey', s=105)

	plt.colorbar(location='top', label=r'$\Sigma({\rm data - model})^2$')
	plt.ylim(6,-0.2)
	plt.xlabel('best-fit single star Teff (K)');plt.ylabel(r'$\Delta{\rm mag}$')

	plt.subplot(326)
	plt.scatter(binary_metrics.cks_teff, binary_metrics.dmag, 
	        c=binary_metrics.chi_sq, cmap='inferno_r',
	       vmin=chi_sq_bins[0],
	       vmax=chi_sq_bins[5], edgecolors='dimgrey', s=105)
	plt.colorbar(location='top', label=r'$\chi^2 = ({\rm data - model})^2/\sigma^2$')
	plt.ylim(6,-0.2)
	plt.xlabel('best-fit single star Teff (K)');plt.ylabel(r'$\Delta{\rm mag}$')

	plt.subplots_adjust(hspace=0.4)
	plt.suptitle(title_str)
	plt.tight_layout()

