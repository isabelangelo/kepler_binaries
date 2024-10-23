from specmatchemp.spectrum import read_hires_fits
from scipy.ndimage import convolve1d
from astropy.table import Table
from astropy.io import fits
from training_utils import *
import matplotlib.pyplot as plt
import specmatchemp.library
import specmatchemp.kernels
import thecannon as tc
import pandas as pd
import numpy as np
import dwt
import os

# =============== load training data =============================================

# path to store training data
df_path = './data/cannon_training_data'

# training labels
lib = specmatchemp.library.read_hdf()
training_set_table = lib.library_params.copy()

# snr cutoff for training set
training_set_table = training_set_table.query('snr>100')

# remove outliers GJ570B + Gl 896A (candidate binaries)
training_set_table = training_set_table[~training_set_table['source_name'].str.contains('GJ570')]
training_set_table = training_set_table[~training_set_table['source_name'].str.contains('Gl 896A')]

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
	if idx in indices_to_insert:
		# Add new rows with vsini=3,5,7km/s
		# also inflate noise so Cannon doesn't learn original errors
		rows_with_broadening.append(broadened_vsini_row(idx, 0)) # add noise
		rows_with_broadening.append(broadened_vsini_row(idx, 3))
		rows_with_broadening.append(broadened_vsini_row(idx, 5))
		rows_with_broadening.append(broadened_vsini_row(idx, 7))
	else:
		rows_with_broadening.append(training_set_table.iloc[idx])

# re-write training table with broadened vsini stars
training_set_table = pd.DataFrame(rows_with_broadening)

def broaden(vsini, spec):
    """
    Applies a broadening kernel to the given spectrum (or error)
    Adopted from specmatch.Match() class from Yee & Petigura 2018
    
    note: if vsini=0, spectrum is not broadened, but its 
    errors are inflated to avoid correlated noise between 
    spectrum + its broadened counterparts

    Args:
        vsini (float): vsini to determine width of broadening
        spec (specmatchemp.spectrum.Spectrum): spectrum to broaden
    Returns:
        broadened (specmatchemp.spectrum.Spectrum): Broadened spectrum
    """

    # copy original spectrum
    broadened_spec = spec.copy()

    # add 1% flux errors
    broadened_spec.serr += 0.01
    broadened_spec.s += np.random.normal(0, 0.01, len(broadened_spec.s))
    
    # broaden if vsini=0
    if vsini != 0:
    	# kernel for line broadening
	    SPEED_OF_LIGHT = 2.99792e5
	    dv = (spec.w[1]-spec.w[0])/spec.w[0]*SPEED_OF_LIGHT
	    n = 151     # fixed number of points in the kernel
	    varr, kernel = specmatchemp.kernels.rot(n, dv, vsini)

	    # broadened spectrum
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
training_set_table['id_starname'] = [i.replace(' ','') for i in training_set_table.source_name]

# split training set into cool, hot stars for piecewise model
# remove stars with broadened vsini from hot star model + giants from cool star model
training_set_table_cool = training_set_table.query('(smemp_teff<5300) & (smemp_logg>4)')
training_set_table_cool.to_csv('./data/label_and_metric_dataframes/training_labels_cool.csv', index=None)
training_set_table_hot = training_set_table.query('(smemp_teff>5200) & (broadened_vsini==False)')
training_set_table_hot.to_csv('./data/label_and_metric_dataframes/training_labels_hot.csv', index=None)

# store columns with labels for training
training_set_hot = Table.from_pandas(
    training_set_table_hot[['smemp_teff', 'smemp_logg', 'smemp_feh', 'smemp_vsini']])
training_set_cool = Table.from_pandas(
    training_set_table_cool[['smemp_teff', 'smemp_logg', 'smemp_feh', 'smemp_vsini']])

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

def single_order_training_data(order_idx, training_set_table, filter_wavelets=True):
	"""
	Stores training flux, sigma for a particular HIRES spectrum order.
	For spectra that require vsini broadening, this is applied
	before wavelet-filtering.

	Args:
		order_idx (int): index corresponding to HIRES order of interested
					(for example, order 1 would be order_idx=0)
		training_set_table (pd.DataFrame): dataframe containing names + labels
					far stars in desired training set.
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

def save_training_data(training_set_table, model_suffix, filter_wavelets=True):
	flux_df = pd.DataFrame()
	sigma_df= pd.DataFrame()
	for order_idx in range(0, 16):
	    flux_df_n, sigma_df_n = single_order_training_data(
	    	order_idx, 
	    	training_set_table,
	    	filter_wavelets=filter_wavelets)
	    flux_df = pd.concat([flux_df, flux_df_n])
	    sigma_df = pd.concat([sigma_df, sigma_df_n])
	flux_df.to_csv('{}/training_flux_{}.csv'.format(df_path, model_suffix), index=False)
	sigma_df.to_csv('{}/training_sigma_{}.csv'.format(df_path, model_suffix), index=False)

# for de-bugging purposes
import time
t0=time.time()

# wavelet-filtered flux + flux errors, hot + cool stars	
save_training_data(training_set_table_hot, 'dwt_hot', filter_wavelets=True)
print('wavelet-filtered training flux and sigma for hot stars saved to .csv files')	
save_training_data(training_set_table_cool, 'dwt_cool', filter_wavelets=True)
print('wavelet-filtered training flux and sigma for cool stars saved to .csv files')

# original flux + flux errors, hot + cool stars		
save_training_data(training_set_table_hot, 'original_hot', filter_wavelets=False)
print('wavelet-filtered training flux and sigma for hot stars saved to .csv files')
save_training_data(training_set_table_cool, 'original_cool', filter_wavelets=False)
print('wavelet-filtered training flux and sigma for cool stars saved to .csv files')
print('total time to load training data = {} seconds'.format(time.time()-t0))

# =============== functions to train model + save validation plots =============================

# file with order stats
order_data_path = './data/cannon_models/rchip_order_stats.csv'

# create file if it doesn't already exist
if os.path.exists(order_data_path)==False:
	empty_order_df = pd.DataFrame({'model': [],'label':[],'bias': [],'rms': []})
	# write the DataFrame to a CSV file
	empty_order_df.to_csv(order_data_path, index=False)

def train_cannon_model(order_numbers, model_suffix, piecewise_component,
	filter_type='dwt'):
	"""
	Trains a Cannon model using all the orders specified in order_numbers

	Args:
	order_numbers (list): order numbers to train on, 1-16 for HIRES r chip
	                    e.g., [1,2,6,15,16]
	model_suffix (str): file ending for Cannon model (for example, 'order4' 
						will save data to ./data/cannon_models/order4/)
	piecewise_component (str): 'hot' or 'cool', determines which subset
						to train the model on (cool for Teff<=5300K, 
						hot for Teff>5300K)
	filter_type (str): if 'dwt', model is trained on wavelet filtered data.
	                   if 'original', model is trained on SpecMatch-Emp output data.

	Returns:
	model (tc.CannonModel): trained Cannon model (also saved to file
						with location determined by model_suffix.)
	"""
	# determine piecewise label dataframe
	if piecewise_component == 'hot':
		training_set = training_set_hot
	elif piecewise_component == 'cool':
		training_set = training_set_cool

	# determine dataframe that contains training data
	flux_df = pd.read_csv('{}/training_flux_{}_{}.csv'.format(
		df_path, filter_type, piecewise_component))
	sigma_df = pd.read_csv('{}/training_sigma_{}_{}.csv'.format(
		df_path, filter_type, piecewise_component))

	# store training flux, sigma for selected orders
	# note: for flux, sigma, we index at 1 to exclude order_number column
	training_flux_df = flux_df[flux_df['order_number'].isin(order_numbers)]
	training_sigma_df = sigma_df[sigma_df['order_number'].isin(order_numbers)]

	normalized_flux = training_flux_df.to_numpy()[:,1:].T
	normalized_sigma = training_sigma_df.to_numpy()[:,1:].T
	normalized_ivar = 1/normalized_sigma**2

	# Create a vectorizer that defines our model form.
	vectorizer = tc.vectorizer.PolynomialVectorizer(
		['smemp_teff', 'smemp_logg', 'smemp_feh','smemp_vsini'], 2)

	# Create the model that will run in parallel using all available cores.
	model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
	                       vectorizer=vectorizer)

	# train and store model
	model_path = './data/cannon_models/{}/'.format(model_suffix)
	model_filename = model_path + '{}_cannon_model.model'.format(piecewise_component)
	model.train()
	print('finished training cannon model')
	model.write(model_filename, include_training_set_spectra=True, overwrite=True)
	print('model written to {}'.format(model_filename))

	return model


def train_and_validate_piecewise_model(order_numbers, model_suffix, filter_type='dwt'):

	# create directory to store models in
	model_path = './data/cannon_models/{}/'.format(model_suffix)
	os.mkdir(model_path)

	# train hot + cool components
	cool_cannon_model = train_cannon_model(
		order_numbers, model_suffix, 'cool', filter_type=filter_type)
	hot_cannon_model = train_cannon_model(
		order_numbers, model_suffix, 'hot', filter_type=filter_type)

	# compute + save Cannon output labels for training set stars
	# using leave-one-out cross-validation
	cannon_label_df = leave20pout_label_df(
		cool_cannon_model, 
		hot_cannon_model, 
		training_set_table_cool, 
		training_set_table_hot, 
		order_numbers)
	df_path = './data/cannon_models/{}/cannon_labels.csv'.format(model_suffix)
	cannon_label_df.to_csv(df_path)

	# create + save one-to-one plot 
	plot_one2one(cannon_label_df, model_suffix)

# ====================== train individual cannon models ============================================

# individual orders, wavelet-filtered
# for order_n in range(1,17):
# 	train_and_validate_piecewise_model([order_n], 'order{}_dwt'.format(order_n))

# # all orders, wavelet-filtered
# train_and_validate_piecewise_model([i for i in range(1,17)], 'all_orders_dwt')

# adopted orders 1-7, wavelet-filtered
train_and_validate_piecewise_model([i for i in range(1,7)], 'orders_1-6_dwt_maxLikelihood')

# # all orders, original
# train_and_validate_piecewise_model([i for i in range(1,17)], 'all_orders_original', filter_type='original')

# # adopted orders 1-7, original
# train_and_validate_piecewise_model([i for i in range(1,7)], 'orders_1-7_original', filter_type='original')









