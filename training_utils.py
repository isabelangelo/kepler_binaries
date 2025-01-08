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

def leave20pout_label_df(cool_cannon_model, hot_cannon_model, cool_label_df, hot_label_df,
    order_numbers):
    """
    Compute Cannon output labels for training set stars using leave-one-out
    cross-validation of piecewise Cannon model with hot + cool components."""

    # define training set labels
    smemp_keys = ['smemp_teff', 'smemp_logg', 'smemp_feh', 'smemp_vsini', 'smemp_psf']
    cannon_keys = [i.replace('smemp', 'cannon') for i in smemp_keys] + ['cannon_rv']
    keys = ['id_starname'] + smemp_keys + cannon_keys + ['fit_logL','fit_model','snr']
    vectorizer = tc.vectorizer.PolynomialVectorizer(smemp_keys, 2)


    def training_set_bins(cannon_model):
        """Bin training set into a given model into
        5 subsets for leave-20%-out validation"""
        n_training = len(cannon_model.training_set_labels)
        test_bins = np.linspace(0, n_training, 6, dtype=int)
        test_bins[-1]= n_training # include remainder in last chunk
        return test_bins
    
    def leave1out_cannon_model(cannon_model, s):
        """Train Cannon model on training set with 20% subset held out"""
        
        # remove left out targets from original training data
        training_set_labels_leave1out = np.delete(cannon_model.training_set_labels, s, 0)
        training_set_leave1out = Table(training_set_labels_leave1out, names=smemp_keys)
        normalized_flux_leave1out = np.delete(cannon_model.training_set_flux, s, 0)
        normalized_ivar_leave1out = np.delete(cannon_model.training_set_ivar, s, 0)

        # train model for cross validation
        model_leave1out = tc.CannonModel(
            training_set_leave1out, 
            normalized_flux_leave1out, 
            normalized_ivar_leave1out,
            vectorizer=vectorizer, 
            regularization=None)
        model_leave1out.train()
        return model_leave1out
    
    def leave1out_label_data(cannon_model, label_df, start_idx, stop_idx, 
                                 cool_cannon_model_leave1out, hot_cannon_model_leave1out):
        """Compute + save labels for objects held out of the training set."""
        # store labels for all held-out objects in model
        cannon_label_data = []
        for spectrum_idx in range(start_idx, stop_idx):
            # load object name from training label dataframe
            spectrum_row = label_df.iloc[spectrum_idx]
            id_starname = spectrum_row.id_starname
            # load object labels, flux, ivar from saved model data
            smemp_labels = cannon_model.training_set_labels[spectrum_idx]
            flux = cannon_model.training_set_flux[spectrum_idx]
            ivar = cannon_model.training_set_ivar[spectrum_idx]
            sigma = 1/np.sqrt(ivar)

            # fit cross validation model to data
            spec = spectrum.Spectrum(
                flux, 
                sigma, 
                order_numbers, 
                cool_cannon_model_leave1out,
                hot_cannon_model_leave1out)
            spec.fit_single_star()
            cannon_labels = spec.fit_cannon_labels         

            # store relevant metrics for dataframe
            values = [spectrum_row.id_starname]+smemp_labels.tolist() + \
                    spec.fit_cannon_labels.tolist() + \
                    [spec.fit_logL, spec.fit_model, spectrum_row.snr]
            cannon_label_data.append(dict(zip(keys, values)))
                
        return cannon_label_data
    
    # bin data from cool + hot model component training sets
    cool_test_bins = training_set_bins(cool_cannon_model)    
    hot_test_bins = training_set_bins(hot_cannon_model)
    
    # perform leave-20%-out cross validation for each bin
    cannon_label_dfs = []    
    for i in range(5):
        
        # define index bounds of left out sample
        cool_start_idx, cool_stop_idx = cool_test_bins[i], cool_test_bins[i+1]
        hot_start_idx, hot_stop_idx = hot_test_bins[i], hot_test_bins[i+1]
        s_cool = slice(cool_start_idx, cool_stop_idx)
        s_hot = slice(hot_start_idx, hot_stop_idx)
    
        print('training model with cool objects {}-{}, hot objects {}-{} held out'.format(
            cool_start_idx, cool_stop_idx, hot_start_idx, hot_stop_idx))
        
        # train models for cross-validation
        cool_cannon_model_leave1out = leave1out_cannon_model(cool_cannon_model, s_cool)
        hot_cannon_model_leave1out = leave1out_cannon_model(hot_cannon_model, s_hot)
        
        # store labels, flux + sigma for left out targets
        cool_label_data_i = leave1out_label_data(
            cool_cannon_model, 
            cool_label_df, 
            cool_start_idx, 
            cool_stop_idx, 
            cool_cannon_model_leave1out, 
            hot_cannon_model_leave1out)
        hot_label_data_i = leave1out_label_data(
            hot_cannon_model, 
            hot_label_df, 
            hot_start_idx, 
            hot_stop_idx, 
            cool_cannon_model_leave1out, 
            hot_cannon_model_leave1out)

        # combine validation data from cool + hot models
        label_df_i = pd.DataFrame(cool_label_data_i + hot_label_data_i)
        label_df_i['model_number'] = i
        cannon_label_dfs.append(label_df_i)
       
    cannon_label_data = pd.concat(cannon_label_dfs)
    return cannon_label_data

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
    plt.figure(figsize=(15,6))
    plt.subplot(231)
    teff_bias, teff_rms = plot_label_one2one(
        cannon_label_df.smemp_teff, 
        cannon_label_df.cannon_teff)
    plt.plot([4000,7000],[4000,7000],'b-')
    plt.xlabel('specmatch library Teff (K)');plt.ylabel('Cannon Teff (K)')

    plt.subplot(232)
    logg_bias, logg_rms = plot_label_one2one(
        cannon_label_df.smemp_logg, 
        cannon_label_df.cannon_logg)
    plt.plot([2.3,5],[2.3,5],'b-')
    plt.xlabel('specmatch library logg (dex)');plt.ylabel('Cannon logg (dex)')

    plt.subplot(233)
    feh_bias, feh_rms = plot_label_one2one(
        cannon_label_df.smemp_feh, 
        cannon_label_df.cannon_feh)
    plt.plot([-1.1,0.6],[-1.1,0.6],'b-')
    plt.xlabel('specmatch library Fe/H (dex)');plt.ylabel('Cannon Fe/H (dex)')

    plt.subplot(234)
    vsini_bias, vsini_rms = plot_label_one2one(
        cannon_label_df.smemp_vsini, 
        cannon_label_df.cannon_vsini)
    plt.plot([0,20],[0,20], 'b-')
    plt.xlabel('specmatch library vsini (km/s)');plt.ylabel('Cannon vsini (km/s)')

    plt.subplot(235)
    psf_bias, psf_rms = plot_label_one2one(
        cannon_label_df.smemp_psf, 
        cannon_label_df.cannon_psf)
    plt.plot([0.5,2],[0.5,2], 'b-')
    plt.xlabel('specmatch library PSF');plt.ylabel('Cannon PSF')

    # save stats to dataframe
    keys = ['model','label','bias','rms']
    order_data = pd.DataFrame(
        (dict(zip(keys, [model_suffix, 'teff', teff_bias, teff_rms])),
        dict(zip(keys, [model_suffix, 'logg', logg_bias, logg_rms])),
        dict(zip(keys, [model_suffix, 'feh', feh_bias, feh_rms])),
        dict(zip(keys, [model_suffix, 'vsini', vsini_bias, vsini_rms]))))
    print(order_data)
    order_data_path = './data/cannon_models/rchip_order_stats.csv'
    existing_order_data = pd.read_csv(order_data_path)
    updated_order_data  = pd.concat(
    		[existing_order_data, order_data])
    updated_order_data.to_csv(order_data_path, index=False)


    # save plot
    figure_path = './data/cannon_models/{}/one2one.png'.format(model_suffix)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print('one-to-one plot saved to saved to {}'.format(figure_path))


