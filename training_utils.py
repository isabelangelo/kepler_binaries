"""
Functions to perform leave-one-out cross-validation of training set
and save stats + Cannon output labels for various piecewise
Cannon models.
"""
import matplotlib.pyplot as plt
import thecannon as tc
import pandas as pd
import numpy as np
import spectrum
from astropy.table import Table


def leave1out_label_df(hot_cannon_model, cool_cannon_model, hot_label_df, cool_label_df,
	order_numbers):
	"""
	Compute Cannon output labels for training set stars using leave-one-out
	cross-validation of piecewise Cannon model with hot + cool components."""

	smemp_keys = ['smemp_teff', 'smemp_logg', 'smemp_feh', 'smemp_vsini']
	cannon_keys = [i.replace('smemp', 'cannon') for i in smemp_keys]
	vectorizer = tc.vectorizer.PolynomialVectorizer(smemp_keys, 2)

	def single_component_labels(piecewise_component):
		"""
		Computes cannon output labels for all stars in training set of
		single component of piecewise model. Needs to be run on hot
		+ cool components individally to validate the full piecewise model.
		"""
		if piecewise_component == 'cool':
		    model_to_validate = cool_cannon_model
		    training_labels_to_validate = cool_label_df
		    piecewise_addition = hot_cannon_model
		if piecewise_component == 'hot':
		    model_to_validate = hot_cannon_model
		    training_labels_to_validate = hot_label_df
		    piecewise_addition = cool_cannon_model
		    
		cannon_label_data = []
		for i in range(len(model_to_validate.training_set_labels))[:3]:

			# store labels, flux + sigma for held out target
			smemp_labels = model_to_validate.training_set_labels[i]
			flux = model_to_validate.training_set_flux[i]
			sigma = 1/np.sqrt(model_to_validate.training_set_ivar[i])

			# remove left out target from training data
			training_set_labels_leave1out = np.delete(model_to_validate.training_set_labels, i, 0)
			training_set_leave1out = Table(training_set_labels_leave1out, names=smemp_keys)
			normalized_flux_leave1out = np.delete(model_to_validate.training_set_flux, i, 0)
			normalized_ivar_leave1out = np.delete(model_to_validate.training_set_ivar, i, 0)

			# train model for cross validation
			model_leave1out = tc.CannonModel(
			    training_set_leave1out, 
			    normalized_flux_leave1out, 
			    normalized_ivar_leave1out,
			    vectorizer=vectorizer, 
			    regularization=None)
			model_leave1out.train()

			# fit cross validation model to data
			spec = spectrum.Spectrum(flux, sigma, order_numbers, model_leave1out, piecewise_addition)
			spec.fit_single_star()

			# store cannon labels + metrics
			row = training_labels_to_validate.iloc[i]
			keys = ['id_starname'] + smemp_keys + cannon_keys + ['snr']
			values = [row.id_starname]+smemp_labels.tolist() + spec.fit_cannon_labels.tolist() + [row.snr]
			cannon_label_data.append(dict(zip(keys, values)))

			return cannon_label_data

	# save leave-one-out labels from hot_cool components to dataframe
	cool_leave1out_labels = single_component_labels('cool')
	hot_leave1out_labels = single_component_labels('hot')
	cannon_label_df = pd.DataFrame(cool_leave1out_labels + hot_leave1out_labels)

	return cannon_label_df

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
    
    print(order_data)
    figure_path = './data/cannon_models/{}/one2one.png'.format(model_suffix)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print('one-to-one plot saved to saved to {}'.format(figure_path))


