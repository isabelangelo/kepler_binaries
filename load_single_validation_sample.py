from specmatchemp.spectrum import read_fits
from astropy.table import Table
from rsync_utils import *
import specmatchemp.library
import pandas as pd
import dwt
import glob
import os

# load single star sample, remove training set stars
raghavan2010_singles = pd.read_csv('./data/literature_data/Raghavan2010_singles_obs_ids.csv')
lib = specmatchemp.library.read_hdf()
training_set_table = lib.library_params.copy().query('snr>100')
singles_in_training_set = raghavan2010_singles.resolvable_name.isin(
	training_set_table.source_name.to_numpy())
raghavan2010_singles = raghavan2010_singles[~singles_in_training_set]
raghavan2010_singles['id_starname'] = [i.replace(' ', '') for i in raghavan2010_singles['resolvable_name']]
raghavan2010_singles['lib_obs'] = [i.replace('j', 'rj') for i in raghavan2010_singles['observation_id']]

# add labels from SPOCS/specmatch
specmatch_psf_results = pd.read_csv('./data/literature_data/specmatch_results.csv')
spocs_labels = Table.read('./data/literature_data/Brewer2016_Table8.fits', format='fits').to_pandas()
spocs_labels['id_starname'] = spocs_labels['Name'].str.decode('utf-8').str.replace(' ', '')
raghavan2010_singles = pd.merge(
	raghavan2010_singles, 
	specmatch_psf_results[['obs','psf']],
	left_on='lib_obs', 
	right_on='obs')
raghavan2010_singles = pd.merge(
	raghavan2010_singles, 
	spocs_labels[['id_starname','Teff','logg','[M/H]','Vsini']], 
	on='id_starname')
columns_to_keep = ['id_starname','lib_obs','Teff','logg','[M/H]', 'Vsini','psf']
raghavan2010_singles = raghavan2010_singles[columns_to_keep].rename(columns=
    {'Teff':'spocs_teff','logg':'spocs_logg', '[M/H]':'spocs_feh',
    'Vsini':'spocs_vsini','psf':'smemp_psd'})
raghavan2010_singles.to_csv('./data/label_and_metric_dataframes/raghavan_single_labels.csv', index=False)

# rsync HIRES spectra of Raghavan 2010 single stars ============================================== 
# add row to match CKS obs_id row
print('copying Raghavan 2010 single sample spectra from cadence')
raghavan2010_singles['obs_id'] = ['r'+i for i in raghavan2010_singles.observation_id]
for index, row in raghavan2010_singles.iterrows():
    obs_ids = [row.observation_id.replace('rj','bj'), row.obs_id, row.obs_id.replace('rj','ij')]
    for obs_id in obs_ids:
        obs_filename = obs_id+'.fits'
        if os.path.exists('./data/raghavan2010_singles_spectra/'+obs_filename):
            print('{} already in ./data/raghavan2010_singles_spectra/'.format(obs_filename))
            pass
        else:
            # write command
            command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/raghavan2010_singles_spectra/{}".format(
                obs_filename,
                obs_filename)
            run_rsync(command)
    print('copied {} b,r,i chip spectra to ./data/raghavan2010_singles_spectra/'.format(row.resolvable_name))


# shift and register spectra with specmatch-emp ========================================================

print('shifting and registering Raghavan 2010 single spectra for binary model validation')
raghavan2010_spectrum_ids = [i[37:-5] for i in glob.glob('./data/raghavan2010_singles_spectra/ij*.fits')]
for spectrum_id in raghavan2010_spectrum_ids:
	input_path = './data/raghavan2010_singles_spectra'
	output_path = './data/raghavan2010_singles_spectra_shifted'
	if os.path.exists(output_path+'/r{}_adj.fits'.format(spectrum_id)):
		print('{} already in ./data/raghavan2010_singles_spectra_shifted/'.format(spectrum_id))
		pass
	else:
		command = 'smemp shift -d {} -o {} {}'.format(
		    input_path, 
		    output_path, 
		    spectrum_id)
		os.system(command)


# store wavelet-filtered fluxes ==================================================================

df_path = './data/spectrum_dataframes'
shifted_path = './data/raghavan2010_singles_spectra_shifted'
print('saving wavelet-filtered flux, sigma to dataframes')
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
for order_n in range(1,17):    
	# lists to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []

	# order index for wavelength re-scaling
	order_idx = order_n - 1

	# get order data for all stars in training set
	for idx, row in raghavan2010_singles.iterrows():
		# load file data
		filename = '{}/{}_adj.fits'.format(
			shifted_path,  
			row.observation_id.replace('j','rj'))
		id_starname_list.append(row.id_starname) # save star name for column
		print(row.id_starname, end=', ')

		# load spectrum from file
		# and resample to unclipped HIRES wavelength scale
		# (since flux, sigma arrays get clipped post-wavelet filtering)
		KOI_spectrum = read_fits(filename)
		rescaled_order = KOI_spectrum.rescale(original_wav_data[order_idx])

		# process for Cannon training
		flux_norm, sigma_norm = dwt.load_spectrum(
		    rescaled_order, 
		    True) # perform wavelet filtering

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
# NOTE: to save original flux, just change filter_wavelets in L90 and run this code.
flux_path = '{}/raghavan_single_flux_dwt.csv'.format(df_path)
sigma_path = '{}/raghavan_single_sigma_dwt.csv'.format(df_path)

flux_df.to_csv(flux_path, index=False)
sigma_df.to_csv(sigma_path, index=False)
print('wavelet-filtered spectra saved to:')
print(flux_path)
print(sigma_path)





