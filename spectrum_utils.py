import os
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
import thecannon as tc
from astropy.io import fits
from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d
from specmatchemp.spectrum import read_hires_fits
from specmatchemp import SPECMATCHDIR

__all__ = ["initial_teff_arr", "flux_weights", "teff2radius", "teff2mass", 
            "speed_of_light_kms", "wav_data", "mask_table_cut"]

# load wavelength data
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
wav_data = fits.open(reference_w_filename)[0].data
original_wav_data = read_hires_fits('./data/cks-spectra/rj122.742.fits').w # KOI-1 r chip file
mask_table = pd.read_csv(os.path.join(SPECMATCHDIR, 'hires_telluric_mask.csv'))
mask_table = mask_table.rename(columns={"order": "order_idx"}) # for consistency between code/specmatch

# compute wavelength limits for masks
mask_table_cut = mask_table.query("chip == 'rj'")
max_v_shift = 30*u.km/u.s # padding to account for RV shifts
minw, maxw = [], []
for idx, row in mask_table_cut.iterrows():
    w_order = original_wav_data[row.order_idx]
    # calculate minimum wavelength + padding for RV shift
    minw_idx = np.floor(w_order[row.minpix])
    minw_idx = (minw_idx*u.angstrom*(1-max_v_shift/c.c)).value
    minw.append(minw_idx)
    # calculate maximum wavelength + padding for RV shift
    if row.maxpix==4021:
        maxw_idx = np.ceil(w_order[-1])
    else:
        maxw_idx = np.ceil(w_order[row.maxpix])
    maxw_idx = (maxw_idx*u.angstrom*(1+max_v_shift/c.c)).value
    maxw.append(maxw_idx)
# save to table
mask_table_cut.insert(4, 'minw', minw)
mask_table_cut.insert(5, 'maxw', maxw)

# inistial Teff values for binary model optimizer
teff_grid = np.arange(4000,10000,2000)
initial_teff_arr = [(x, y) for x in teff_grid for y in teff_grid if x>=y]

# speed of light for wavelength calculation
speed_of_light_kms = c.c.to(u.km/u.s).value

# temperature to radius conversion for binary model
pm2013 = pd.read_csv('./data/literature_data/PecautMamajek_table.csv', 
                     skiprows=22, delim_whitespace=True).replace('...',np.nan)
teff_pm2013 = np.array([float(i) for i in pm2013['Teff']])
R_pm2013 = np.array([float(i) for i in pm2013['R_Rsun']])
mass_pm2013 = np.array([float(i) for i in pm2013['Msun']])

valid_mass = ~np.isnan(mass_pm2013)
teff2radius = interp1d(teff_pm2013[valid_mass], R_pm2013[valid_mass])
teff2mass = interp1d(teff_pm2013[valid_mass], mass_pm2013[valid_mass])
mass2teff = interp1d(mass_pm2013[valid_mass], teff_pm2013[valid_mass])

####### functions to calculate binary model + associated chi-squared

def flux_weights(teff1, teff2, wav):
    """Returns un-normalized relative fluxes,
    based on blackbody curve * R^2
    """
    # blackbody functions
    bb1 = BlackBody(temperature=teff1*u.K)
    bb2 = BlackBody(temperature=teff2*u.K)
    # evaluate blackbody at model wavelengths
    bb1_curve = bb1(wav*u.AA).value
    bb2_curve = bb2(wav*u.AA).value

    # calculate unweighted flux contributions
    B1 = bb1_curve*teff2radius(teff1)**2
    B2 = bb2_curve*teff2radius(teff2)**2
    
    # normalize weights to sum to 1
    B_sum = B1 + B2
    W1 = B1/B_sum
    W2 = B2/B_sum

    return W1, W2


