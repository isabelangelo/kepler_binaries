import pandas as pandas
import thecannon as tc
from spectrum import *

# hot + cool Cannon model components
# (for final version, these should be the cleaned models)
cool_cannon_model = tc.CannonModel.read('./data/cannon_models/orders_1-6_dwt/cool_cannon_model.model')
hot_cannon_model = tc.CannonModel.read('./data/cannon_models/orders_1-6_dwt/hot_cannon_model.model')

extracted_order_numbers = [i for i in range(1,7)]
def load_spectrum_df(path):
	"""Load spectrum dataframe and extract orders of interest for analysis."""
	spectrum_df = pd.read_csv(path)
	extracted_orders_df = spectrum_df[spectrum_df['order_number'].isin(extracted_order_numbers)]
	return extracted_orders_df

# flux of single star sample
single_flux_df = load_spectrum_df('./data/spectrum_dataframes/raghavan_single_flux_dwt.csv')
single_sigma_df = load_spectrum_df('./data/spectrum_dataframes/raghavan_single_sigma_dwt.csv')

# flux of binary sample
binary_flux_df = load_spectrum_df('./data/spectrum_dataframes/kraus_sullivan_binary_flux_dwt.csv')
binary_sigma_df = load_spectrum_df('./data/spectrum_dataframes/kraus_sullivan_binary_sigma_dwt.csv')

# metrics to store for spectra in validation samples
single_keys = ['id_starname', 'cannon_teff', 'cannnon_logg','cannon_feh','cannon_vsini', 'cannon_rv',\
               'fit_logL', 'fit_model','fit_BIC']
binary_keys = ['cannon_teff1', 'cannnon_logg1','cannon_feh12','cannon_vsini1', 'cannon_rv1', \
               'cannon_teff2', 'cannnon_logg2','cannon_vsini2', 'cannon_rv2', \
               'binary_fit_logL', 'binary_fit_model','binary_BIC', \
               'delta_BIC','binary_dRV','binary_teff_ratio']

def metric_df(flux_df, sigma_df):
	"""
	Compute best-fit single star + binary labels and binary detection methods
	for all stars in sample.
	Args:
		flux_df (pd.DataFrame): dataframe containing fluxes of sample of interest,
								with first column indicating order_number (should only
								include pixels in trained Cannon model orders)
								(N_pixels x (1+N_stars)),
		sigma_df (pd.DataFrame): same as flux_df, but with flux errors.
	"""
	metric_data = []
	for star in flux_df.columns[1:].to_numpy():
	    spec = Spectrum(flux_df[star], sigma_df[star], 
	                    order_numbers, cool_cannon_model, hot_cannon_model)
	    spec.fit_single_star()
	    spec.fit_binary()
	    spec.fit_binary_drv = spec.binary_fit_cannon_labels[4] - spec.binary_fit_cannon_labels[-1]
	    spec.fit_binary_teff_ratio = spec.binary_fit_cannon_labels[5]/spec.binary_fit_cannon_labels[0]
	    single_metrics = [star]+list(spec.fit_cannon_labels) + \
	                    [spec.fit_logL, spec.fit_model, spec.fit_BIC]
	    binary_metrics = list(spec.binary_fit_cannon_labels) + \
	                    [spec.binary_fit_logL, spec.binary_fit_model, spec.binary_fit_BIC, \
	                     spec.delta_BIC, spec.fit_binary_drv, spec.fit_binary_teff_ratio]
	    metric_data.append(dict(zip(single_keys+binary_keys, single_metrics+binary_metrics))) 

	metric_df = pd.DataFrame(metric_data)
	return metric_df


# load metrics from single star sample  
single_metric_df = metric_df(single_flux_df, single_sigma_df)

# load metrics for binary sample
binary_metric_df = metric_df(binary_flux_df, binary_sigma_df)
