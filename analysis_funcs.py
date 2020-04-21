"""
Functions for analysis of spectral data and behavioural response
"""
#import scipy as sp
import numpy as np
import xarray
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging
import seaborn
import pandas as pd
import pymc3 as pm
from pandas.api.types import CategoricalDtype
import copy

def read_spectra(src=os.getcwd(),tag_with_folder=True):
    """ reads in all spectra in a directory, using the folder name as the spectrum_type
    tag in the attrs field
    
    Parameters
    ----------
    src : str
        src directory
        
    tag_with_folder : bool
        if True, add folder name as attrs tag
            
    """
    spectra = []
    flist = os.listdir(src)
    # retains all files 
    flist = [f for f in flist if f.endswith('csv') or f.endswith('DTI')]
    for fname in flist:
        try:
            raw = _read_csv(os.path.join(src,fname))
        except:
            try:
                raw = _read_dti(os.path.join(src,fname))
            except:
                logging.warn('{} could not be read'.format(os.path.join(src,fname)))
                continue
                #raise
            
        spec = xarray.DataArray(raw[:,1],dims=['wavelength'],coords=[raw[:,0]])
        spec = spec.interp(wavelength=np.arange(2000))
        # regrid to 1nm and append
        if tag_with_folder:
            spec.attrs['spectrum_type'] = os.path.relpath(src).split('/')[-1]
            spec.attrs['source_file'] = os.path.join(src,fname)
            spec.attrs['spectrum_name'] = fname.split('.')[-2]
        spectra.append(spec)
    return spectra

def _read_csv(fpath):
    """reads a spectral format csv"""
    # conversion dict for non standard encoding
    convs = {0:lambda s: float(s[1:]),1:lambda s: float(s[1:])}
    try:
        return np.loadtxt(fpath,delimiter=';',encoding='utf')
    except:
        try:
            return np.loadtxt(fpath,delimiter=',',encoding='utf')
        except:
            return np.loadtxt(fpath,delimiter=';',encoding='utf',converters=convs)

def _read_dti(fpath):
    arr = np.loadtxt(fpath,skiprows=3)
    return np.stack([arr[::2],arr[1::2]]).T
    
def p_response(spectrum,photoreceptors):
    """
    Returns the integrated response of a list of photoreceptors for a given spectrum
    
    Parameters
    ----------
    spectrum : xarray.DataArray
        spectrum xarray with wavelength coordinates
        
    photoreceptors : list
        list of xarray.DataArray objects with wavelength coordinates
        
    """
    responses = []
    response_spec = []
    for pr in photoreceptors:
        res = spectrum*pr
        response_spec.append(res)
        responses += [res.fillna(0).integrate('wavelength').to_dict()['data']]
    return np.array(responses), response_spec

class lightSimulator(object):
    """
    Class for taking measured spectra and returning
    predictions for different PWM values
    
    Parameters
    ----------
    spectrum : xarray.DataArray
        measured spectrum of LED array
    """
    def __init__(self,spectrum,pwm_values=[255,120,255,240],
                 cut_points=[0,410,490,600,900]):
        self.leds = []
        self.cps = cut_points
        for i,pwm in enumerate(pwm_values):
            self.leds.append(spectrum[self.cps[i]:self.cps[i+1]]/pwm)
        
    
    def predict(self,new_pwm_values):
        if len(new_pwm_values) != len(self.leds):
            raise ValueError('pwm value should be of length {}'.format(len(self.leds)))
        out = xarray.DataArray(np.empty(self.cps[-1]-self.cps[0]),
                               dims=['wavelength'],
                               coords = [np.arange(self.cps[0],self.cps[-1])])
        for i, pwm in enumerate(new_pwm_values):
            out[self.cps[i]:self.cps[i+1]] = self.leds[i]*int(pwm)
        out = out.fillna(0)
        out.attrs = self.leds[0].attrs
        out.attrs['PWM_values'] = new_pwm_values
        return out
    
def pwm_to_normalized_response(pwms,reference_pwm,reference_spectrum,
                               photoreceptors,normalisation_spectrum=None):
    """
    Converts PWM settings on calibrated light source to insect visual response
    normalised to a normalisation_spectrum
    
    Parameters:
    -----------
    pwms : list
        list of lists of LED PWM values to be converted
    
    reference_pwm : list
        list of LED PWM values for reference spectrum
    
    reference_spectrum : xarray.DataArray
        reference spectrum xarray with wavelength coordinates
    
    photoreceptors : list
        list of xarray.DataArray objects with wavelength coordinates
        for each photoreceptor
    
    normalisation_spectrum : xarray.DataArray
        optional spectrum to normalise responses to
    """
        
    ls = lightSimulator(reference_spectrum,reference_pwm)
    norm_resp,_ = p_response(normalisation_spectrum,photoreceptors)
    # normalise to 1
    norm_resp = norm_resp/norm_resp.sum()
    out = []
    for pwm in pwms:
        resp,_ = p_response(ls.predict(pwm),photoreceptors)
        if 'normalisation_spectrum' in dir():
            resp = resp/resp.sum()
            normed = resp/norm_resp
            out.append(normed/normed.sum())
            
        else:
            out.append(resp)
    
    return np.array(out), norm_resp

def response_barplot(response,title='Response Plot'):
    ax = seaborn.barplot(['short','mid','long'],response/response.sum())
    plt.hlines(.3333,-.5,2.5,linestyles='dashed')
    # set individual bar lables using above list
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.1, i.get_height()+.01, \
                '{:.2f}'.format(i.get_height()/total), fontsize=15,
                    color='dimgrey')
    plt.title(title)
    plt.xlabel('Receptor')
    plt.ylabel('Relative Response')
    # adjust y padding
    plt.ylim((0,.7))
    return ax

def monochromatic_boundary(photoreceptors,min_wlength=300,max_wlength=800):
    """
    Computes the monochromatic relative response for every wavelength between
    min_wlength and max_wlength
    """
    wlengths = np.arange(min_wlength,max_wlength)
    results=np.empty((len(wlengths)-1,len(photoreceptors)))
    
    for i in range(len(wlengths)-1):
        v,_ = p_response(xarray.DataArray(data=[100,100],
                                          dims=['wavelength'],
                                          coords=[[wlengths[i],wlengths[i+1]]]),photoreceptors)
        results[i,:] = v
        
    out = np.empty((len(wlengths)-1 ,len(photoreceptors)))
    for i in range(len(photoreceptors)):
        out[:,i] = results[:,i]/results.sum(axis=1)
        
    return out

def monochromatic_lims(mc_boundary,min_wlength=300,max_wlength=800):
    """
    Find the maximum colour coordinates in each axis
    """
    wlengths = np.arange(min_wlength,max_wlength)
    
    out = np.empty((mc_boundary.shape[1],mc_boundary.shape[1]))
    wlengths_out = np.empty(mc_boundary.shape[1])
    for i in range(mc_boundary.shape[1]):
        out[i,:] = mc_boundary[np.nanargmax(mc_boundary[:,i]),:]
        wlengths_out[i] = wlengths[np.nanargmax(mc_boundary[:,i])]
    return out,wlengths_out.astype(int)

    
def generate_o3(response,amplitude,cx,cy,latex=False):
    #
    params = [response,amplitude,cx,cy]
    # generate all components
    # 2 2-way
    if latex:
        return [
        '${0} \sim ({1}*{3})+({1}*{2})$'.format(*params), # 0
        # 1 2-way
        '${0} \sim ({1}*{2})+({3})$'.format(*params), # 1
        '${0} \sim ({1}*{3})+({2})$'.format(*params), # 2
        # 1 2-way
        '${0} \sim ({1}*{2})$'.format(*params), # 3
        '${0} \sim ({1}*{3})$'.format(*params), # 4
        # additive 3 features
        '${0} \sim {1}+{2}+{3}$'.format(*params), # 5
        # additive 2 features
        '${0} \sim {1}+{2}$'.format(*params), # 6
        '${0} \sim {1}+{3}$'.format(*params), # 7
        '${0} \sim {2}+{3}$'.format(*params), # 8
        # 
        '${0} \sim {1}$'.format(*params), # 9
        '${0} \sim {2}$'.format(*params), # 10
        '${0} \sim {3}$'.format(*params)] # 11
    else:
        return [
        '{0} ~ ({1}*{3})+({1}*{2})'.format(*params), # 0
        # 1 2-way
        '{0} ~ ({1}*{2})+({3})'.format(*params), # 1
        '{0} ~ ({1}*{3})+({2})'.format(*params), # 2
        # 1 2-way
        '{0} ~ ({1}*{2})'.format(*params), # 3
        '{0} ~ ({1}*{3})'.format(*params), # 4
        # additive 3 features
        '{0} ~ {1}+{2}+{3}'.format(*params), # 5
        # additive 2 features
        '{0} ~ {1}+{2}'.format(*params), # 6
        '{0} ~ {1}+{3}'.format(*params), # 7
        '{0} ~ {2}+{3}'.format(*params), # 8
        # 
        '{0} ~ {1}'.format(*params), # 9
        '{0} ~ {2}'.format(*params), # 10
        '{0} ~ {3}'.format(*params)] # 11
    
def run_models(df,formulae,nitt=3000,tune=600):
    models = []
    traces = []
    for fr in formulae:
        with pm.Model() as logistic_model:
            pm.glm.GLM.from_formula(fr,
                                    df,
                                    family=pm.glm.families.Binomial())
            trace = pm.sample(nitt, tune=tune, init='adapt_diag',progressbar=True)
        traces.append(trace)
        models.append(logistic_model)
    return models, traces

def plot_traces(traces, retain=0):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:],
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')

def lm_pred_additive(coefs, new_grid):
    """
    
    PARAMETERS
    ----------
    coefs : dict
        dictionary of coefficients
        
    new_grid : dict
        k, v for every k in coefs where
        v is a 1D array-like of predict values
    """
    
    
    if len(new_grid) == len(coefs)-1:
        shapes = [x.shape[0] for x in new_grid.values()]
        XX = np.array(np.meshgrid(*new_grid.values()))
        XX = XX.reshape((len(new_grid),-1))
        odds_ = np.ones(XX.shape[1])*coefs['Intercept']

        for i,coef in enumerate(new_grid.keys()):
            odds_ += coefs[coef] * XX[i,:]

        odds_ = np.exp(odds_).reshape(shapes)
        #return odds_
        
        return odds_ / (1 + odds_)
    
    else:
        raise ValueError('coefs and new_grid sizes don\'t match')
        
        
## to get predictions, easiest is to sample from posterior traces
def predict_from_trace(trace, new_grid, n_samples=2000):
    """
    Takes a trace object, new parameter values and samples
    from the traces to estimate the distribution of values in
    data space. Works only for models with fixed intercept.
    Don't supply intercept in new_grid
    """
    # get idxs of sample
    idxs = np.arange(len(trace['Intercept'])//trace.nchains)
    np.random.shuffle(idxs)
    idxs = idxs[:n_samples]
    
    shape = []
    # get out shape
    for k in new_grid.keys():
        if k != 'Intercept':
            shape.append(len(new_grid[k]))
    shape.append(n_samples)
    
    out = np.empty(shape)
    
    # get coefs
    for i, j in enumerate(idxs):
        out[...,i] = lm_pred_additive(trace[j], new_grid)
    return out

def estimate_1d(X, cred_int = 95):
    """
    Estimate the predicted response from a sampled
    array of predictions with trace in last dimension
    """
    est = X.mean(axis=-1)
    lower = np.percentile(X, (100-cred_int)/2, axis=-1)
    upper = np.percentile(X, cred_int+((100-cred_int)/2), axis=-1)
    return est, lower, upper


def plot_extras(points,**kwargs):
    """
    takes a list of dicts with matplotlib.pyplot.plot keywords
    """
    ax = plt.gca()
    for trace in points:
        ax.scatter(**trace,**kwargs)


def plot_cx_cy(prediction, grid, ax=None, **kwargs):
    """
    Takes the prediction dictionary and generates a plot
    
    PARAMETERS
    ----------
    prediction : dict
        output dict from predict_from_trace func
    
    grid : dict
        new grid specified to predict_from_trace
    """
    if ax==None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
    if prediction.ndim > 3:
        raise ValueError('More than 2 dimensions in posterior predictive')
    
    preds = prediction.mean(axis=-1)
    i = ax.imshow(preds,aspect='auto',origin='lower',
              extent=[grid['cx'].min(),grid['cx'].max(),grid['cy'].min(),grid['cy'].max()],
              **kwargs)
    plt.colorbar(i)
    CS = ax.contour(grid['cx'],grid['cy'],preds,[.4,.5,.6,.7,.8],colors='k')
    ax.clabel(CS,inline=1, fontsize=10,colors='k')
    
    return ax

def prob_pos_neg(trace_arr):
    """
    Returns probability of pos, probability of neg for a trace
    
    PARAMETERS
    ----------
    trace_arr : array-like
        array of sampled values
    """
    total = len(trace_arr)
    pr_pos = np.sum(trace_arr>0)/total
    pr_neg = np.sum(trace_arr<0)/total
    print('Pr_positive: {}, Pr_negative: {}'.format(pr_pos,pr_neg))

def compare(c1,c2,inv_link=np.exp):
    return -((inv_link(c1)-inv_link(c2))/inv_link(c1))*100

def inv_logit(x):
    return np.exp(x)/(1+np.exp(x))

def ppc(trace,model,df,cats=['Cultivar','LT'], samples=1000):
    # copy index to separate id col
    df['id'] = df.index

    # set index to multilevel treatment combo
    df.index = [df[x] for x in cats]

    # get unique treatment combos
    unique_treatments=np.unique(df.index)

    # get list of indexes for unique treatments
    unique_IDs = [df['id'][x].iloc[0] for x in unique_treatments]

    # change index back again
    df.index = df.id
    
    # sample posterior predictive
    preds = pm.sample_posterior_predictive(trace, model=model, samples=samples)
    
    # get groupings
    groups = {}
    for i, c in enumerate(cats):
        groups[c] = pd.Categorical(np.repeat([x[i] for x in unique_treatments],samples),
                                   categories=np.unique(df[c]),ordered=False).reorder_categories(df[c].cat.categories)
        
    
    groups['PosteriorPredictive'] = (preds['y'][:,unique_IDs]).T.ravel()
    
def transmission(f,solar):
    """
    Returns UVB,UVA,PAR transmission
    """
    PAR = (f*solar)[400:700].fillna(0).integrate('wavelength')/solar[400:700].fillna(0).integrate('wavelength')
    UVA = (f*solar)[315:400].fillna(0).integrate('wavelength')/solar[315:400].fillna(0).integrate('wavelength')
    UVB = (f*solar)[280:315].fillna(0).integrate('wavelength')/solar[280:315].fillna(0).integrate('wavelength')
    return UVB*100,UVA*100,PAR*100