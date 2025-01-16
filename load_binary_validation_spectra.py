from specmatchemp.spectrum import read_fits
from rsync_utils import *
import specmatchemp.library
import pandas as pd
import dwt
import glob
import os

# unresolved binary sample
kraus_unresolved_companions = pd.read_csv('./data/literature_data/Kraus_KOI_sample/kraus_unresolved_companions.csv')
kraus_unresolved_companions = kraus_unresolved_companions.rename(columns={"Name": "resolvable_name"})

kraus_binary_obs_ids = pd.read_csv('./data/literature_data/Kraus_unresolved_binary_obs_ids.csv')
kraus_binary_id_starnames = [i.replace('KOI-', 'K0') for i in kraus_binary_obs_ids.resolvable_name]
kraus_binary_obs_ids.insert(1, 'id_starname', kraus_binary_id_starnames)

cks_labels = pd.read_csv('./data/label_and_metric_dataframes/cks_labels.csv')

# add binary detection information from Kraus catalog
kraus_binary_labels = pd.merge(kraus_binary_obs_ids[['resolvable_name', 'id_starname', 'observation_id']], 
         kraus_unresolved_companions[['resolvable_name','CC','Pbin', 'sep_arcsec','sep_mas','dmag','dKmag','source']], 
         on='resolvable_name')
# Replace NaN values in the 'dmag' column with the corresponding 'dKmag' values
kraus_binary_labels.loc[kraus_binary_labels['dmag'].isna(), 'dmag'] = kraus_binary_labels['dKmag']

# flag stars with multiple unresolved companions
# note: I won't use the CC designation, since this includes companions with sep>0.8"
kraus_binary_labels['multiple_companions'] = False
duplicates = kraus_binary_labels['id_starname'].duplicated(keep=False)
kraus_binary_labels['multiple_companions'] = duplicates

# remove fainter companions
kraus_binary_labels = kraus_binary_labels.sort_values(by=['id_starname', 'dmag'], ascending=[True, True])
kraus_binary_labels = kraus_binary_labels.drop_duplicates(subset='id_starname', keep='first')

# if reported, include CKS labels
kraus_binary_labels = pd.merge(
	kraus_binary_labels, 
	cks_labels[['id_starname','cks_teff','cks_logg','cks_feh','cks_vsini']], 
	on='id_starname', how='left')
kraus_binary_labels.to_csv('./data/label_and_metric_dataframes/kraus_binary_labels.csv')


# # rsync HIRES spectra of Raghavan 2010 single stars ============================================== 
# # add row to match CKS obs_id row
# print('copying Kraus unrsolved binary sample spectra from cadence')
# kraus_unresolved_binaries['obs_id'] = ['r'+i for i in kraus_unresolved_binaries.observation_id]
# for index, row in kraus_unresolved_binaries.iterrows():
#     obs_ids = [row.observation_id.replace('rj','bj'), row.obs_id, row.obs_id.replace('rj','ij')]
#     for obs_id in obs_ids:
#         obs_filename = obs_id+'.fits'
#         if os.path.exists('./data/kraus_unresolved_binary_spectra/'+obs_filename):
#             print('{} already in ./data/kraus_unresolved_binary_spectra/'.format(obs_filename))
#             pass
#         else:
#             # write command
#             command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/kraus_unresolved_binary_spectra/{}".format(
#                 obs_filename,
#                 obs_filename)
#             run_rsync(command)
#     print('copied {} b,r,i chip spectra to ./data/kraus_unresolved_binary_spectra/'.format(row.resolvable_name))


# # shift and register spectra with specmatch-emp ========================================================

# print('shifting and registering Kraus unrsolved binary spectra for binary model validation')
# kraus_spectrum_ids = [i[40:-5] for i in glob.glob('./data/kraus_unresolved_binary_spectra/ij*.fits')]
# for spectrum_id in kraus_spectrum_ids:
# 	input_path = './data/kraus_unresolved_binary_spectra'
# 	output_path = './data/kraus_unresolved_binary_spectra_shifted'
# 	if os.path.exists(output_path+'/r{}_adj.fits'.format(spectrum_id)):
# 		print('{} already in ./data/kraus_unresolved_binary_spectra_shifted/'.format(spectrum_id))
# 		pass
# 	else:
# 		command = 'smemp shift -d {} -o {} {}'.format(
# 		    input_path, 
# 		    output_path, 
# 		    spectrum_id)
# 		os.system(command)


# store wavelet-filtered fluxes ==================================================================

df_path = './data/spectrum_dataframes'
shifted_path = './data/kraus_unresolved_binary_spectra_shifted'
print('saving wavelet-filtered flux, sigma to dataframes')

def flux_and_sigma_dataframes(filter_wavelets = True):
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
		for idx, row in kraus_binary_labels.iterrows():
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
			    filter_wavelets) # if True, performs wavelet filtering

			# save to lists
			flux_list.append(flux_norm)
			sigma_list.append(sigma_norm)

		# store flux, sigma data
		flux_df_n = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
		sigma_df_n = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

		# store order number
		flux_df_n.insert(0, 'order_number', order_n)
		sigma_df_n.insert(0, 'order_number', order_n)

		# generate final dataframe
		flux_df = pd.concat([flux_df, flux_df_n])
		sigma_df = pd.concat([sigma_df, sigma_df_n])

	return flux_df, sigma_df

# write wavelet-filtered flux, sigma to .csv files
flux_df_dwt, sigma_df_dwt = flux_and_sigma_dataframes(filter_wavelets = True)
flux_path_dwt = '{}/kraus_binary_flux_dwt.csv'.format(df_path)
sigma_path_dwt = '{}/kraus_binary_sigma_dwt.csv'.format(df_path)
flux_df_dwt.to_csv(flux_path_dwt, index=False)
sigma_df_dwt.to_csv(sigma_path_dwt, index=False)
print('wavelet-filtered spectra saved to:')
print(flux_path_dwt)
print(sigma_path_dwt)

# write original flux, sigma to .csv files
flux_df_original, sigma_df_original = flux_and_sigma_dataframes(filter_wavelets = False)
flux_path_original = '{}/kraus_binary_flux_original.csv'.format(df_path)
sigma_path_original = '{}/kraus_binary_sigma_original.csv'.format(df_path)
flux_df_original.to_csv(flux_path_original, index=False)
sigma_df_original.to_csv(sigma_path_original, index=False)
print('original spectra saved to:')
print(flux_path_original)
print(sigma_path_original)







