from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pywt
#import specmatchemp.library

# define tranform function inputs
wt_kwargs = {
'mode':'constant', # extends signal based on edge values
'axis':-1 # axis over which to compute the DWT
}

# clip size to clip end of order (in pixels)
# set to 390 for r chip to ensure orders don't overlap
# set to 200 for i chip, correpsonding to 5% on each side
order_clip = 390

def flux_waverec(flux, wavelet_wavedec, coeff_indices_to_keep):
	"""
	compute reconstructed spectrum from coefficients output 
	from the spectrum's wavelet decomposition. 

	Args:
		flux (np.array): normalized flux for decomposition (must have even # of elements)
		wavelet_wavedec (str): name of mother wavelet in decomposition/reconstruction 
								('sym2', 'haar', etc.)
		coeff_indices_to_keep (list): list of indexes (e.g., [1,2,3]) corresponding to 
						coefficient arrays to preserve in the wavelet recombination. 
						For indices not in this list,the coefficient arrays will be set to 
						zero before the wavelet recombination.
						NOTE: the coefficient list is arranged as [cAn, cDn, cDn-1, …, cD2, cD1] 
						where n is the number of levels, so index 0 corresponds to the 
						approximation coefficients, index 1 corresponds to the highest 
						decomposition level,  and index -1 corresponds to the lowest level.
	Returns:
		flux_waverec (np.array): normalized reconstructed flux, including only decomposition
								levels specified by 'levels'
	"""
	# wavelet decomposition
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	coeffs = pywt.wavedec(flux, wavelet_wavedec, level = max_level, **wt_kwargs)
	# set coefficients to zero for all other levels
	for idx in range(0, max_level+1): # iterate over all levels, including last
		if idx not in coeff_indices_to_keep:
			coeffs[idx] = np.zeros_like(coeffs[idx])
	# wavelet reconstruction
	flux_waverec = pywt.waverec(coeffs, wavelet_wavedec, **wt_kwargs)
	return flux_waverec

def load_spectrum(rescaled_order, filter_wavelets):
	"""
	Load flux, sigma of a spectrum from a .fits file.
	These flux + sigma values can be used during the Cannon
	training and test steps.
	Args:
		filename (str): path to shifted + registered HIRES spectrum
		filter_wavelets (bool): if True, perform wavelet-based filtering to spectrum
								if False, return original spectrum
	"""

	# store flux, sigma
	flux_norm = rescaled_order.s
	sigma_norm = rescaled_order.serr
	w_order = rescaled_order.w

	# remove nans from flux, sigma
	# note: this needs to happen here so that the Cannon
	# always returns flux values for all wavelengths
	finite_idx = ~np.isnan(flux_norm)
	if np.sum(finite_idx) != len(flux_norm):
		flux_norm = np.interp(w_order, w_order[finite_idx], flux_norm[finite_idx])
	sigma_norm = np.nan_to_num(sigma_norm, nan=1)

	# require even number of elements
	if len(flux_norm) %2 != 0:
		flux_norm = flux_norm[:-1]
		sigma_norm = sigma_norm[:-1]

	if filter_wavelets:
		# compute wavelet transform of flux
		idx_min, idx_max = 1,8 # coefficient indices to preserve in transform
		coeff_idx = np.arange(idx_min,idx_max+1,1)
		flux_rec = flux_waverec(flux_norm, 'sym5', coeff_idx)
		#flux_rec += 1 # normalize to 1 for training
		flux_norm = flux_rec

	# clip order on each end
	flux_norm = flux_norm[order_clip:-1*order_clip]
	sigma_norm = sigma_norm[order_clip:-1*order_clip]

	return flux_norm, sigma_norm


def plot_flux_waverec_residuals(flux, wavelet_wavedec, object_name):
	"""
	Plot the original spectrum and reconstructed spectrum from coefficients output 
	from the spectrum's wavelet decomposition at each individual level, 
	along with residuals between the two. This is to test how much information 
	is lost in the wavelet decomposition - small residuals indicate 
	that most infromation is preserved.

	Args:
		flux (np.array): normalized flux for decomposition (must have even # of elements)
		wavelet_wavedec (str): name of mother wavelet in decomposition/reconstruction 
								('sym2', 'haar', etc.)
		object_name (str): object identifier to go in the plot title (e.g., 'CK00367')
	"""
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	all_level_flux_waverec = flux_waverec(
		flux, 
		wavelet_wavedec, 
		np.arange(0, max_level+1))
	diff = flux - all_level_flux_waverec
	print(np.mean(abs(diff)))
	plt.figure(figsize=(15,5))
	plt.subplot(211);plt.title(object_name)
	plt.plot(w, flux, color='k', label='original spectrum')
	plt.plot(w, all_level_flux_waverec, 'r--', label='IDWT(wavelet coefficients)')
	plt.ylabel('normalized flux');plt.legend(frameon=False)
	plt.subplot(212)
	plt.plot(w, diff, 'k-')
	plt.xlabel('wavelength (angstrom)')
	plt.ylabel('residuals')
	path = './figures/{}_waverec_residuals.png'.format(wavelet_wavedec)
	#plt.savefig(path, dpi=150)


def plot_flux_waverec_levels(w, flux, wavelet_wavedec, object_name):
	"""
	Plot the reconstructed spectrum from coefficients output 
	from the spectrum's wavelet decomposition at each individual level.

	Args:
		flux (np.array): normalized flux for decomposition (must have even # of elements)
		wavelet_wavedec (str): name of mother wavelet in decomposition/reconstruction 
								('sym2', 'haar', etc.)
		object_name (str): object identifier to go in the plot title (e.g., 'CK00367')
	"""
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	n_levels = max_level + 1
	fig, axes = plt.subplots(n_levels+1, 1, sharex=True, sharey=False, 
		figsize=(7,8), tight_layout=True)
	plt.rcParams['font.size']=8
	axes[0].plot(w, flux, color='k', lw=0.5)
	#axes[0].text(6760, 0.1, 'original signal')
	#axes[0].set_ylim(-0.6,0.4)
	for level in range(0, n_levels):
		level_flux_waverec = flux_waverec(
			flux, 
			wavelet_wavedec,
			[level])
		axes[n_levels - level].plot(w, level_flux_waverec, 'k-', lw=0.5)
		#axes[n_levels - level].set_title('level = {}'.format(level), fontsize=8)
	fig.suptitle(object_name)
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')
	plt.subplots_adjust(hspace=0.4)
	#plt.savefig('/Users/isabelangelo/Desktop/wavelet_levels_2019_bior22.png')
	plt.show()

# plot the difference between 2019 and 2022
def plot_waverec_level_diff(w, wavelet_wavedec):
	# wavedec_idx = (w>5197) & (w<5288)
	flux_2019 = fits.open(
	    './data/kepler1656_spectra/all_orders/CK00367_2019_rj351.570_adj_resampled.fits')[1].data['s']
	flux_2022 = fits.open(
	    './data/kepler1656_spectra/all_orders/CK00367_2022_rj487.76_adj_resampled.fits')[1].data['s']

	max_level = pywt.dwt_max_level(len(flux_2019), wavelet_wavedec)
	fig, axes = plt.subplots(max_level+1, 2, sharex=True, sharey=False, 
	                         figsize=(14,14), tight_layout=True)

	for level in range(max_level+1):
	    level_2019 = flux_waverec(flux_2019, wavelet_wavedec,[level])
	    level_2022 = flux_waverec(flux_2022, wavelet_wavedec,[level])
	    axis_n = max_level-level
	    axes[axis_n, 0].plot(w, level_2019, color='orangered', alpha=0.7)
	    axes[axis_n, 0].plot(w, level_2022, color='cornflowerblue', alpha=0.7)
	    axes[axis_n, 0].set_ylim(-0.2,0.2)
	    axes[axis_n, 0].text(6715, 0.1, 'level = {}'.format(level))
	    if level == max_level:
	        axes[axis_n, 0].text(6670, 0.07, '2019', color='orangered')
	        axes[axis_n, 0].text(6670, -0.15, '2022', color='cornflowerblue')
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')


	for level in range(max_level+1):
	    level_2019 = flux_waverec(flux_2019, wavelet_wavedec, [level])
	    level_2022 = flux_waverec(flux_2022, wavelet_wavedec, [level])
	    level_diff = level_2019 - level_2022
	    rms = "{:0.2e}".format(np.sqrt(np.sum(level_diff**2)/len(level_diff)))
	    axis_n = max_level-level
	    axes[axis_n, 1].plot(w, level_diff, 'k-', alpha=0.7)
	    axes[axis_n, 1].set_ylim(-0.05,0.05)
	    axes[axis_n, 1].text(6705, 0.03, 'level = {}, rms = {}'.format(level, rms), color='firebrick')
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')

	axes[0,0].set_title('IDWT of orginal signal\n at each decomposition level', pad=30)
	axes[0,1].set_title('IDWT difference (2019 - 2022)', pad=30)
	fig.suptitle('wavelet = {}'.format(wavelet_wavedec))
	path = './figures/{}_level_diff.png'.format(wavelet_wavedec)
	#plt.savefig(path, dpi=150)


