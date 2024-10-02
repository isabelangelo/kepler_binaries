from specmatchemp.spectrum import read_hires_fits
from scipy.ndimage import convolve1d
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import specmatchemp.library
import specmatchemp.kernels
import thecannon as tc
import pandas as pd
import numpy as np
import spectrum
import dwt
import os

# =============== load training data =============================================

# path to store training data
df_path = './data/cannon_training_data'

# training labels
lib = specmatchemp.library.read_hdf()
training_set_table = lib.library_params.copy()

# snr cutoff for training set
training_set_table = training_set_table.query('snr>150')

# TEMPORARY: temperature cutoff to test slow rotators
training_set_table = training_set_table.query('Teff<=5300')
# remove outlier GJ570B (candidate binary)
training_set_table = training_set_table[~training_set_table['source_name'].str.contains('GJ570')]

# =============== handle vsini upper limits =========================================

# add ladder of broadened spectra for rows with vsini upper limits
# assume vsini=0 and record indices
training_set_table['vsini'] = training_set_table['vsini'].replace('--',0.0) 
training_set_table['broadened_vsini'] = False # for all targets pre-broadening
indices_to_insert = np.where(training_set_table.vsini==0)[0]

# insert new rows where vsini=0 with broadened vsini values
def broadened_vsini_row(idx, vsini):
    new_vsini_row = training_set_table.iloc[idx].copy()
    new_vsini_row['vsini']=vsini
    new_vsini_row['broadened_vsini']=True
    new_vsini_row['source_name'] = new_vsini_row['source_name']+'-'+str(vsini)
    return new_vsini_row
rows_with_broadening = []
for idx in range(len(training_set_table)):
	new_row = training_set_table.iloc[idx] # original target information

	if idx in indices_to_insert:
		# Add new rows with vsini=3,5,7km/s
		rows_with_broadening.append(broadened_vsini_row(idx, 3))
		rows_with_broadening.append(broadened_vsini_row(idx, 5))
		rows_with_broadening.append(broadened_vsini_row(idx, 7))
		# rename current row to flag vsini=0
		new_row['source_name'] = new_row['source_name']+'-0'

	# Keep the original row
	rows_with_broadening.append(new_row)  
# re-write training table with broadened vsini stars
training_set_table = pd.DataFrame(rows_with_broadening)

def broaden(vsini, spec):
    """
    Applies a broadening kernel to the given spectrum (or error)
    Adopted from specmatch.Match() class from Yee & Petigura 2018

    Args:
        vsini (float): vsini to determine width of broadening
        spec (Spectrum): spectrum to broaden
    Returns:
        broadened (Spectrum): Broadened spectrum
    """
    SPEED_OF_LIGHT = 2.99792e5
    dv = (spec.w[1]-spec.w[0])/spec.w[0]*SPEED_OF_LIGHT
    n = 151     # fixed number of points in the kernel
    varr, kernel = specmatchemp.kernels.rot(n, dv, vsini)
    # broadened = signal.fftconvolve(spec, kernel, mode='same')

    broadened_spec = spec.copy()
    broadened_spec.s = convolve1d(broadened_spec.s, kernel)
    broadened_spec.serr = convolve1d(broadened_spec.serr, kernel)

    return broadened_spec

# =============== save training set + wavelength data =============================================

# rewrite columns and save to file
training_set_table = training_set_table.rename(columns=
    {'Teff':'smemp_teff','logg':'smemp_logg', 'feh':'smemp_feh','vsini':'smemp_vsini'})
training_set_table = training_set_table.astype(
    {'smemp_teff': 'float32', 
     'smemp_logg': 'float32', 
     'smemp_feh': 'float32', 
     'smemp_vsini': 'float32'})
training_set = Table.from_pandas(
    training_set_table[['smemp_teff', 'smemp_logg', 'smemp_feh', 'smemp_vsini']])
training_set_table['id_starname'] = [i.replace(' ','') for i in training_set_table.source_name]
training_set_table.to_csv('./data/label_and_metric_dataframes/training_labels.csv', index=None)

# write clipped wavelength data to reference file
# (this is used to rescale specmatch spectra 
# into individual orders for wavelet-filtering)
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') # KOI-1 original r chip file
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements
wav_data = original_wav_data[:, dwt.order_clip:-1*dwt.order_clip] # clip 5% on each side
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
fits.HDUList([fits.PrimaryHDU(wav_data)]).writeto(reference_w_filename, overwrite=True)
print('clipped reference wavlength saved to {}'.format(reference_w_filename))

# =============== load and save training data =============================================

def single_order_training_data(order_idx, filter_wavelets=True):
	"""
	Stores training flux, sigma for a particular HIRES spectrum order.
	For spectra that require vsini broadening, this is applied
	before wavelet-filtering.

	Args:
		order_idx (int): index corresponding to HIRES order of interested
					(for example, order 1 would be order_idx=0)
		filter_wavelets (bool): if True, performs wavelet-based filtering
					on flux of order of interested. If false, stores flux
					and sigma of original, continuum-normalized 
					flux from specmatch-emp library.

	Returns:
		flux_df_n (pd.DataFrame): flux for training set for order of interest, 
					with n_rows=number of pixels in order, n_cols = number of 
					training set objects
		sigma_df_n (pd.DataFrame): same as flux_df_n, but contains flux errors.

	"""
	# order numbers are not zero-indexed
	order_n = order_idx + 1

	# places to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []

	# get order data for all stars in training set
	#for i in range(len(training_set_table)):
	for idx, row in training_set_table.iterrows():

	    # load file data
	    #row = training_set_table.iloc[i]
	    id_starname_list.append(row.source_name.replace(' ','')) # save star name for column

	    # load spectrum from specmatch-emp library
	    # and resample to unclipped HIRES wavelength scale
	    # (since flux, sigma arrays get clipped post-wavelet filtering)
	    KOI_spectrum = lib.get_spectrum(row.lib_index)
	    rescaled_order = KOI_spectrum.rescale(original_wav_data[order_idx])

	    # process for Cannon training
	    # (includes vsini broadening and wavelet-filtering)
	    if row.broadened_vsini==True:
	    	flux_norm, sigma_norm = dwt.load_spectrum(
	    		broaden(row.smemp_vsini, rescaled_order), 
	    		filter_wavelets)
	    else:
	    	flux_norm, sigma_norm = dwt.load_spectrum(
		        rescaled_order, 
		        filter_wavelets)

	    # save to lists
	    flux_list.append(flux_norm)
	    sigma_list.append(sigma_norm)

	# store flux, sigma data
	flux_df_n = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
	sigma_df_n = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

	# store order number
	flux_df_n.insert(0, 'order_number', order_n)
	sigma_df_n.insert(0, 'order_number', order_n)

	return flux_df_n, sigma_df_n

# for de-bugging purposes
import time
t0=time.time()

# wavelet-filtered flux + flux errors
training_flux_dwt = pd.DataFrame()
training_sigma_dwt = pd.DataFrame()
for order_idx in range(0, 16):
    flux_dwt_n, sigma_dwt_n = single_order_training_data(order_idx)
    training_flux_dwt = pd.concat([training_flux_dwt, flux_dwt_n])
    training_sigma_dwt = pd.concat([training_sigma_dwt, sigma_dwt_n])
training_flux_dwt.to_csv('{}/training_flux_dwt.csv'.format(df_path), index=False)
training_sigma_dwt.to_csv('{}/training_sigma_dwt.csv'.format(df_path), index=False)
print('wavelet-filtered training flux and sigma saved to .csv files')

# original flux + flux errors
training_flux_original = pd.DataFrame()
training_sigma_original = pd.DataFrame()
for order_idx in range(0, 16):
    flux_original_n, sigma_original_n = single_order_training_data(order_idx, filter_wavelets=False)
    training_flux_original = pd.concat([training_flux_original, flux_original_n])
    training_sigma_original = pd.concat([training_sigma_original, sigma_original_n])
training_flux_original.to_csv('{}/training_flux_original.csv'.format(df_path), index=False)
training_sigma_original.to_csv('{}/training_sigma_original.csv'.format(df_path), index=False)
print('training flux and sigma pre wavelet filter saved to .csv files')
print('total time to load training data = {} seconds'.format(time.time()-t0))

# =============== functions to train model + save validation plots =============================

# file with order stats
order_data_path = './data/cannon_models/rchip_order_stats.csv'

# create file if it doesn't already exist
if os.path.exists(order_data_path)==False:
	empty_order_df = pd.DataFrame({'model': [],'label':[],'bias': [],'rms': []})
	# write the DataFrame to a CSV file
	empty_order_df.to_csv(order_data_path, index=False)


def plot_label_one2one(x, y):
	"""
	Computes the RMS and bias associated with the Cannon-inferred
	stellar labels, used to generate one-to-one plots.
	"""
	diff = y - x
	bias = np.round(np.mean(diff), 3)
	rms = np.round(np.sqrt(np.sum(diff**2)/len(diff)), 3)
	plt.plot(x, y, 'b.')
	plt.plot([], [], '.', color='w', 
			label = 'rms = {}\nbias = {}'.format(rms, bias))
	plt.legend(loc='upper left', frameon=False, labelcolor='b')
	return bias, rms

def plot_one2one(cannon_label_df, model_suffix):
	"""
	Generates a one-to-one plot of the known training labels
	and Cannon-inferred labels for the training vallidation sample,
	as computed by a particular Cannon model of interest.
	"""
	plt.figure(figsize=(15,3))
	plt.subplot(141)
	teff_bias, teff_rms = plot_label_one2one(
		cannon_label_df.smemp_teff, 
		cannon_label_df.cannon_teff)
	plt.plot([4000,7000],[4000,7000],'b-')
	plt.xlabel('specmatch library Teff (K)');plt.ylabel('Cannon Teff (K)')

	plt.subplot(142)
	logg_bias, logg_rms = plot_label_one2one(
		cannon_label_df.smemp_logg, 
		cannon_label_df.cannon_logg)
	plt.plot([2.3,5],[2.3,5],'b-')
	plt.xlabel('specmatch library logg (dex)');plt.ylabel('Cannon logg (dex)')

	plt.subplot(143)
	feh_bias, feh_rms = plot_label_one2one(
		cannon_label_df.smemp_feh, 
		cannon_label_df.cannon_feh)
	plt.plot([-1.1,0.6],[-1.1,0.6],'b-')
	plt.xlabel('specmatch library Fe/H (dex)');plt.ylabel('Cannon Fe/H (dex)')

	plt.subplot(144)
	vsini_bias, vsini_rms = plot_label_one2one(
		cannon_label_df.smemp_vsini, 
		cannon_label_df.cannon_vsini)
	plt.plot([0,20],[0,20], 'b-')
	plt.xlabel('specmatch library vsini (km/s)');plt.ylabel('Cannon vsini (km/s)')

	# save stats to dataframe
	keys = ['model','label','bias','rms']
	order_data = pd.DataFrame(
		(dict(zip(keys, [model_suffix, 'teff', teff_bias, teff_rms])),
		dict(zip(keys, [model_suffix, 'logg', logg_bias, logg_rms])),
		dict(zip(keys, [model_suffix, 'feh', feh_bias, feh_rms])),
		dict(zip(keys, [model_suffix, 'vsini', vsini_bias, vsini_rms]))))
	existing_order_data = pd.read_csv(order_data_path)
	updated_order_data  = pd.concat(
			[existing_order_data, order_data])
	updated_order_data.to_csv(order_data_path, index=False)

def compute_training_cannon_labels(cannon_model, order_numbers, filter_type):
	"""
	Computes Cannon-inferred stellar labels for training set using the 
	Cannon's test step. 
	note: This is not a formal model validation so it doesn't use 
	leave-one-out validation, but it may be updated in the future
	to perform leave-one-out.

	Args:
		cannon_model (tc.CannonModel): cannon model of interest
		order_numbers (list): HIRES chip order numbers model was 
				trained on (e.g., [1,2,4,5]).
	"""
	# names of keys + metrics to store
	labels_to_plot = ['teff', 'logg', 'feh', 'vsini']
	smemp_keys = ['smemp_'+i for i in labels_to_plot]
	cannon_keys = [i.replace('smemp', 'cannon') for i in smemp_keys]
	keys = ['id_starname'] + smemp_keys + cannon_keys + ['fit_chisq','training_density']

	# load training flux, sigma at orders of interest
	if filter_type=='dwt':
		training_flux = training_flux_dwt[training_flux_dwt['order_number'].isin(order_numbers)]
		training_sigma = training_sigma_dwt[training_sigma_dwt['order_number'].isin(order_numbers)]
	elif filter_type=='original':
		training_flux = training_flux_original[training_flux_original['order_number'].isin(order_numbers)]
		training_sigma = training_sigma_original[training_sigma_original['order_number'].isin(order_numbers)]

	# compute cannon labels for training stars + store to dataframe
	cannon_label_data = []
	for idx, row in training_set_table.iterrows():
	    spec = spectrum.Spectrum(
	            training_flux[row.id_starname], 
	            training_sigma[row.id_starname], 
	            order_numbers, 
	            cannon_model)
	    spec.fit_single_star()
	    values = [row.id_starname] + row[smemp_keys].values.tolist() \
	            + spec.fit_cannon_labels.tolist() + [spec.fit_chisq, spec.training_density]
	    cannon_label_data.append(dict(zip(keys, values)))

	# convert label data to dataframe
	cannon_label_df = pd.DataFrame(cannon_label_data)
	return cannon_label_df

def train_cannon_model(order_numbers, model_suffix, filter_type='dwt', 
	save_training_data=False):
	"""
	Trains a Cannon model using all the orders specified in order_numbers
	order_numbers (list): order numbers to train on, 1-16 for HIRES r chip
	                    e.g., [1,2,6,15,16]
	model_suffix (str): file ending for Cannon model (for example, 'order4' 
						will save data to ./data/cannon_models/order4/)
	filter_type (str): if 'dwt', model is trained on wavelet filtered data.
	                   if 'original', model is trained on SpecMatch-Emp output data.
	save_training_data (bool): if True, saves dataframes of training flux + sigma
	                    to ./data/cannon_training_data
	"""

	# determine dataframe that contains training data
	if filter_type=='dwt':
	    flux_df = training_flux_dwt
	    sigma_df = training_sigma_dwt
	else:
	    flux_df = training_flux_original
	    sigma_df = training_sigma_original

	# store training flux, sigma for selected orders
	# note: for flux, sigma, we index at 1 to exclude order_number column
	training_flux_df = flux_df[flux_df['order_number'].isin(order_numbers)]
	training_sigma_df = sigma_df[sigma_df['order_number'].isin(order_numbers)]
	normalized_flux = training_flux_df.to_numpy()[:,1:].T
	normalized_sigma = training_sigma_df.to_numpy()[:,1:].T
	normalized_ivar = 1/normalized_sigma**2

	# save training data to a .csv
	if save_training_data:
	    flux_path = '{}training_flux_{}.csv'.format(
	        training_data_path,model_suffix)
	    sigma_path = '{}training_sigma_{}.csv'.format(
	        training_data_path,model_suffix)
	    training_flux_df.to_csv(flux_path, index=False)
	    training_sigma_df.to_csv(sigma_path, index=False)

	# Create a vectorizer that defines our model form.
	vectorizer = tc.vectorizer.PolynomialVectorizer(
		['smemp_teff', 'smemp_logg', 'smemp_feh','smemp_vsini'], 2)

	# Create the model that will run in parallel using all available cores.
	model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
	                       vectorizer=vectorizer)

	# train and store model
	model_path = './data/cannon_models/{}/'.format(model_suffix)
	os.mkdir(model_path)
	model_filename = model_path + 'cannon_model.model'
	model.train()
	print('finished training cannon model')
	model.write(model_filename, include_training_set_spectra=True, overwrite=True)
	print('model written to {}'.format(model_filename))

	# save one-to-one plot
	cannon_label_df = compute_training_cannon_labels(model, order_numbers, filter_type)
	cannon_label_filename = model_path + 'cannon_labels.csv'
	cannon_label_df.to_csv(cannon_label_filename, index=False)
	print('cannon labels saved to {}'.format(cannon_label_filename))

	# generate one-to-one plots
	print('generating one-to-one diagnostic plots of training set')
	plot_one2one(cannon_label_df, model_suffix)
	figure_path = model_path + 'one2one.png'
	print('one-to-one plot saved to saved to {}'.format(figure_path))
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')

 # ====================== train individual cannon models ============================================

# for testing purposes
for order_n in range(1, 2):
	#train_cannon_model([order_n], 'order{}_dwt_nan_vsini_broadened'.format(order_n))
	train_cannon_model([order_n], 'order{}_dwt_nan_vsini_broadened_cool'.format(order_n))
	#train_cannon_model([order_n], 'order{}_original_nan_vsini_broadened'.format(order_n), filter_type='original')

# to do: I need to fix the bug that fits the dwt spectra to the original model.
# I think it's fixed, I'll run it to test.
# oh but now the difference is that I'm using my code
# and not the test step. so it might be slightly different but shouldn't differ by much.
# should I do this in the jupyter notebook just to see?
# maybe  while I'm waiting?








