"""
Loads CKS sample flux + labels
for single star Cannon model validation
"""
from specmatchemp.spectrum import read_fits
from astropy.table import Table
from astropy.io import fits
from rsync_utils import *
import specmatchemp.library
import dwt
import pandas as pd
import numpy as np
import pexpect
import glob
import os

# =================== generate table with CKS labels =========================

# table with CKS stars
cks_main_stars_path = './data/literature_data/Petigura2017_Table5.fits'
cks_main_stars = Table(fits.open(cks_main_stars_path)[1].data).to_pandas()
print(len(cks_main_stars), 'stars from CKS')

# remove KOI-2864, which seems to have some RV pipeline processing errors
cks_main_stars = cks_main_stars[~cks_main_stars.Name.str.contains('KOI-02864')]
print(len(cks_main_stars), ' after removing KOI-02864 due to processing errors')

# table with CKS-cool stars
cks_cool_stars = pd.read_csv('./data/literature_data/CKS-cool_Spectroscopic_Parameters.csv')
cks_cool_stars['id_obs'] = cks_cool_stars['id_obs'].apply(lambda x: 'r' + x) # rename to match cks stars
min_teff = 4200 # lowest Teff where Cannon quadratic assumption holds
max_teff = cks_main_stars['Teff'].min()
cks_cool_stars = cks_cool_stars.query('smsyn_teff<@max_teff & smsyn_teff>@min_teff')
print('{} stars from CKS-cool with Teff={}-{}K'.format(
    len(cks_cool_stars), min_teff, max_teff))

# rename columns of CKS sample
cks_cols_to_keep = ['Name', 'Obs','Teff', 'e_Teff', 'logg', 'e_logg', \
                    '[Fe/H]', 'e_[Fe/H]','vsini', 'e_vsini']
cks_main_stars = cks_main_stars[cks_cols_to_keep].rename(
    columns={
    "Name": "id_starname", 
    "Obs": "obs_id",
    "Teff": "cks_teff",
    "e_Teff": "cks_teff_err",
    "logg":"cks_logg",
    "e_logg": "cks_logg_err", 
    "[Fe/H]": "cks_feh",
    "e_[Fe/H]": "cks_feh_err",
    "vsini": "cks_vsini",
    "e_vsini": "cks_vsini_err"})
cks_main_stars['sample'] = ['cks'] * len(cks_main_stars)
# re-format star names to be consistent with filenames
cks_main_stars.id_starname = [i.replace('KOI-', 'K').replace(' ', '') for i in cks_main_stars.id_starname]

# rename columns of CKS-cool
cks_cool_cols_to_keep = ['id_name', 'id_obs','smsyn_teff', 'smsyn_teff_err', 'smsyn_logg', \
        'smsyn_logg_err', 'smsyn_fe', 'smsyn_fe_err','smsyn_vsini']
cks_cool_stars = cks_cool_stars[cks_cool_cols_to_keep].rename(
    columns={
    "id_name": "id_starname", 
    "id_obs": "obs_id",
    "smsyn_teff": "cks_teff",
    "smsyn_teff_err": "cks_teff_err",
    "smsyn_logg": "cks_logg", 
    "smsyn_logg_err": "cks_logg_err",
    "smsyn_fe": "cks_feh",
    "smsyn_fe_err": "cks_feh_err",
    "smsyn_vsini": "cks_vsini"})
cks_cool_stars['sample'] = ['cks-cool'] * len(cks_cool_stars)

# combine samples for training set
cks_stars = pd.concat([cks_main_stars, cks_cool_stars], ignore_index=True)
# remove stars with vsini>=11km/s (upper limit of specmatch training set)
cks_stars = cks_stars.query('cks_vsini<11')
# re-format obs ids
cks_stars.obs_id = [i.replace(' ','') for i in cks_stars.obs_id]

# save to file
cks_stars_filename = './data/label_and_metric_dataframes/cks_labels.csv'
cks_stars.to_csv(cks_stars_filename, index=False)
print('table with CKS + CKS-cool stars ({} total) saved to {}'.format(
    len(cks_stars),
    cks_stars_filename))

# =================== transfer spectra from cadence =========================

# # copy over CKS stars
# for index, row in cks_stars.iterrows():
#     # filenames for rsync command
#     obs_ids = [row.obs_id.replace('rj','bj'), row.obs_id, row.obs_id.replace('rj','ij')]
#     for obs_id in obs_ids:
#         obs_filename = obs_id+'.fits'
#         if os.path.exists('./data/cks-spectra/'+obs_filename):
#             print('{} already in ./data/cks-spectra/'.format(obs_filename))
#             pass
#         else:
#             # write command
#             command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/cks-spectra/{}".format(
#                 obs_filename,
#                 obs_filename)
#             run_rsync(command)
#     print('copied {} b,r,i chip spectra to ./data/cks-spectra/'.format(row.id_starname))


# # ================== shift + register spectra =============================================
# spectrum_ids = [i[20:-5] for i in glob.glob('./data/cks-spectra/ij*.fits')]
# for spectrum_id in spectrum_ids:
#     input_path = './data/cks-spectra'
#     output_path = './data/cks-spectra_shifted'
#     command = 'smemp shift -d {} -o {} {}'.format(
#         input_path, 
#         output_path, 
#         spectrum_id)
#     os.system(command)

# ================== wavelet-filter + store spectra =============================================

# store flux, sigma for all orders with both original and filtered flux
def cks_sample_data(filter_wavelets):
    flux_df = pd.DataFrame()
    sigma_df = pd.DataFrame()
    for order_n in np.arange(1,17,1):    
        # lists to store data
        id_starname_list = []
        flux_list = []
        sigma_list = []
        # order index for wavelength re-scaling
        order_idx = order_n - 1

        # get order data for all stars in training set
        for i in range(len(cks_stars)):
            # load file data
            row = cks_stars.iloc[i]
            filename = './data/cks-spectra_shifted/{}_adj.fits'.format( 
                row.obs_id)
            id_starname = row.id_starname.replace(' ', '')
            id_starname_list.append(id_starname) # save star name for column
            print(id_starname)

            # load spectrum from file
            # and resample to unclipped HIRES wavelength scale
            # (since flux, sigma arrays get clipped post-wavelet filtering)
            KOI_spectrum = read_fits(filename)
            rescaled_order = KOI_spectrum.rescale(original_wav_data[order_idx])

            # process for Cannon training, save to lists
            flux_norm, sigma_norm = dwt.load_spectrum(
                rescaled_order, 
                filter_wavelets) # set to True to perform wavelet-filtering
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
        return flux_df, sigma_df

# write flux, sigma to .csv files
print('storing flux, sigma of CKS sample to dataframes')
df_path = './data/spectrum_dataframes'
flux_df_dwt, sigma_df_dwt = cks_sample_data(True)
flux_df_dwt.to_csv('{}/cks_flux_dwt.csv'.format(df_path), index=False)
sigma_df_dwt.to_csv('{}/cks_sigma_dwt.csv'.format(df_path), index=False)
print('wavelet-filtered CKS spectra saved to .csv files')
flux_df_original, sigma_df_original = cks_sample_data(False)
flux_df_original.to_csv('{}/cks_flux_original.csv'.format(df_path), index=False)
sigma_df_original.to_csv('{}/cks_sigma_original.csv'.format(df_path), index=False)
print('original CKS spectra saved to .csv files')
