"""
Loads labels + HIRES spectra for the binary validation sets.

(1) saves spectra + metrics for all binaries from Kraus 2016 + Kolbl 2015
(2) sorts binaries into three categories based on which Cannon model should perform best:
	- "binary": Teff2=4200-7000K (training set range), binary model should perform best
	- "single": Teff2<3300 (below noise floor), single model should perform best
	- "neither": Teff2=3300-4200K, neither model should perform well

"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.spectrum import read_fits
from spectrum import Spectrum
import spectrum_utils
import thecannon as tc
import pandas as pd
import numpy as np
import dwt


# ============== load binaries from Kraus 2016, Kolbl 2015 ==================================================

# define paths to spectrum files + labels
df_path = './data/spectrum_dataframes'
shifted_path = './data/cks-spectra_shifted'

# path to names + labels of Kraus 2016 + Kolbl 2015 binaries
kraus2016_binaries = pd.read_csv('./data/label_dataframes/kraus2016_binary_labels.csv')
kolbl2015_binaries = pd.read_csv('./data/label_dataframes/kolbl2015_binary_labels.csv')
known_binaries = pd.concat((
	kraus2016_binaries[['id_starname', 'obs_id']], 
	kolbl2015_binaries[['id_starname', 'obs_id']]))

# filter fluxes with wavelet decomposition
filter_wavelets=True
# store orders in relevant Cannon model
order_numbers = [i for i in np.arange(1,17,1).tolist() if i not in [8, 11, 16]]
model_path = './data/cannon_models/rchip/adopted_orders_dwt'
cannon_model = tc.CannonModel.read(model_path+'/cannon_model.model')

# load original wavelength data for rescaling
# this is from the KOI-1 original r chip file
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') 
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements

# store flux, sigma for all orders
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
print('storing flux, sigma of binaries to dataframes')
for order_n in order_numbers:    
	# lists to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []
	# order index for wavelength re-scaling
	order_idx = order_n - 1

	# get order data for all stars in training set
	for i in range(len(known_binaries)):
		# load file data
		row = known_binaries.iloc[i]
		filename = '{}/{}_adj.fits'.format(
            shifted_path,  
            row.obs_id) # .replace('rj','ij')) # for i chip
		id_starname_list.append(row.id_starname) # save star name for column

		# load spectrum from file
		# and resample to unclipped HIRES wavelength scale
		# (since flux, sigma arrays get clipped post-wavelet filtering)
		KOI_spectrum = read_fits(filename)
		rescaled_order = KOI_spectrum.rescale(original_wav_data[order_idx])

		# process for Cannon training
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

	# save to final dataframe
	flux_df = pd.concat([flux_df, flux_df_n])
	sigma_df = pd.concat([sigma_df, sigma_df_n])

# write flux, sigma to .csv files
flux_path = '{}/known_binary_flux_dwt.csv'.format(df_path)
sigma_path = '{}/known_binary_sigma_dwt.csv'.format(df_path)

flux_df.to_csv(flux_path, index=False)
sigma_df.to_csv(sigma_path, index=False)
print('wavelet-filtered binary spectra saved to:')
print(flux_path)
print(sigma_path)

# compute metrics for binary sample and save to dataframe
print('computing metrics for binary sample:')
metric_keys = ['fit_chisq', 'training_density', 'binary_fit_chisq', 'delta_chisq', 'delta_BIC', 'f_imp']
binary_label_keys = ['teff1', 'logg1', 'feh12', 'vsini1', 'rv1', 'teff2', 'logg2', 'vsini2' , 'rv2']
binary_label_keys = ['cannon_'+i for i in binary_label_keys]
metric_data = []
for star in flux_df.columns[1:]:
	print(star)
	# load flux, sigma
	flux = flux_df[star]
	sigma = sigma_df[star]
	# create spectrum object
	spec = Spectrum(
		flux, 
		sigma, 
		order_numbers, 
		cannon_model)
	# calculate metrics
	spec.fit_single_star()
	spec.fit_binary()
	# store metrics in dataframe
	keys = ['id_starname'] + metric_keys + binary_label_keys
	values = [star] + [spec.fit_chisq, spec.training_density, \
	spec.binary_fit_chisq, spec.delta_chisq, spec.delta_BIC, spec.f_imp] + \
	spec.binary_fit_cannon_labels.tolist()
	metric_data.append(dict(zip(keys, values)))
# convert metric data to dataframe
metric_df = pd.DataFrame(metric_data)
metric_path = './data/metric_dataframes/known_binary_metrics.csv'
metric_df.to_csv(metric_path)
print('known binary metrics saved to {}'.format(metric_path))
print('')


# ============== sort binaries based on secondary Teff ========================================================

print('sorting binaries into categories based on secondary Teff..')
# sort Kolbl companions based on their temperatures
kolbl2015_companions = kolbl2015_binaries.rename(columns={'Teff_A':'teff1', 'Teff_B':'teff2'})
kolbl2015_companions['companion_type'] = np.nan
kolbl2015_companions.loc[(kolbl2015_companions.teff2>=4200),'companion_type'] = 'A'
kolbl2015_companions.loc[(kolbl2015_companions.teff2>=3000) & (kolbl2015_companions.teff2<4200),'companion_type'] = 'B'
kolbl2015_companions.loc[(kolbl2015_companions.teff2<3000),'companion_type'] = 'C'

# sort the kraus companions based on their temperature
# note: I am loading the original table here because I need the m2 information for 
# every companion in cases where a single star has multiple companions
kraus2016_companions = pd.read_csv('./data/literature_data/Kraus2016/Kraus2016_Table7.csv', delim_whitespace=True)
kraus2016_companions['m1'] = kraus2016_companions['m2']/kraus2016_companions['q']
kraus2016_companions['teff1'] = [spectrum_utils.mass2teff(i).item() for i in kraus2016_companions['m1']]
kraus2016_companions['teff2'] = [spectrum_utils.mass2teff(i).item() for i in kraus2016_companions['m2']]
kraus2016_companions['companion_type'] = np.nan
kraus2016_companions.loc[(kraus2016_companions.teff2>=4200),'companion_type'] = 'A'
kraus2016_companions.loc[(kraus2016_companions.teff2>=3000) & (kraus2016_companions.teff2<4200),'companion_type'] = 'B'
kraus2016_companions.loc[(kraus2016_companions.teff2<3000),'companion_type'] = 'C'

# sort binaries into whether they are well-fit by binary model, 
# single star model, or neither
metric_df.insert(1, 'model_type', np.nan)
binary_data = []

for id_starname in metric_df.id_starname:
    
    # names for cross-matching across datasets
    kolbl_koi = int(id_starname.replace('K0',''))
    kraus_koi = id_starname.replace('K0', 'KOI-')

    # lists to store companion information
    pairs = []; pair_info = []
    
    # companions from Kolbl 2015
    if kolbl_koi in kolbl2015_companions.KOI.to_numpy():
        companions_df = kolbl2015_companions[kolbl2015_companions.KOI==kolbl_koi]
        for i in range(len(companions_df)):
            row = companions_df.iloc[i]
            pairs.append(row.companion_type)
            pair_info.append((id_starname, int(row.teff1), int(row.teff2)))
    else:
        pass
    
    # companions from Kraus 2016
    if kraus_koi in kraus2016_companions.KOI.to_numpy():
        companions_df = kraus2016_companions[kraus2016_companions.KOI==kraus_koi]
        for i in range(len(companions_df)):
            row = companions_df.iloc[i]
            pairs.append(row.companion_type)
            pair_info.append((id_starname, int(row.teff1), int(row.teff2)))
    else:
        pass

    # sorting pairs into category
    pairs = list(set(pairs))
    pair_info = list(set(pair_info))
    if len(pairs)==0:
        model_type = 'NaN'
    elif pairs.count('A')==1 and pairs.count('B')==0:
        model_type = 'binary'
    elif pairs.count('B')>0:
        model_type = 'neither'
    elif pairs.count('A')==0 and pairs.count('B')==0:
        model_type = 'single'
    else:
        print('NEEDS CATEGORY')
    
    for pair in pair_info:
        keys = ['id_starname', 'teff1','teff2', 'model_type']
        values = list(pair) + [model_type]
        binary_data.append(dict(zip(keys, values)))
    # add type to binary metric dataframe
    metric_df.loc[metric_df.id_starname==id_starname, 'model_type'] = model_type
    # store companion data to compute mass ratios
    companion_df = pd.DataFrame(binary_data)

# sort binaries into categories and store
single_model_binaries = metric_df.query('model_type=="single"')
binary_model_binaries = metric_df.query('model_type=="binary"')
neither_model_binaries = metric_df.query('model_type=="neither"')

print(len(metric_df), ' stars with 1+ companion in sample')
print(len(single_model_binaries), ' should be well-fit by the single star model (companions are too faint, Teff<3000K)')
print(len(binary_model_binaries), ' should be well-fit by binary model (just one companion with Teff=4200-7000K)')
print(len(neither_model_binaries), ' should poorly fit by both models (1+ star with Teff=3000-4200K)')

# add mass ratio values to binaries that The Cannon can model
binary_model_binaries.insert(5, 'q_true', np.nan)
binary_model_binaries.insert(6, 'q_cannon', np.nan)
for id_starname in binary_model_binaries.id_starname:
    companions = companion_df[companion_df.id_starname==id_starname]
    cannon_companions = companions.query('teff2>=4200').iloc[0]
    m2 = spectrum_utils.teff2mass(cannon_companions.teff2)
    m1 = spectrum_utils.teff2mass(cannon_companions.teff1)
    binary_model_binaries.loc[binary_model_binaries.id_starname==id_starname, 'q_true'] = m2/m1
    
for i in range(len(binary_model_binaries)):
    row =  binary_model_binaries.iloc[i]
    m2 = spectrum_utils.teff2mass(row.cannon_teff2).item()
    m1 = spectrum_utils.teff2mass(row.cannon_teff1).item()
    binary_model_binaries.loc[binary_model_binaries.id_starname==row.id_starname, 'q_cannon'] = m2/m1

# store to dataframes
single_model_binaries.to_csv('./data/metric_dataframes/single_model_binary_metrics.csv')
binary_model_binaries.to_csv('./data/metric_dataframes/binary_model_binary_metrics.csv')
neither_model_binaries.to_csv('./data/metric_dataframes/neither_model_binary_metrics.csv')

print('categories saved to separate dataframes')
print('')

