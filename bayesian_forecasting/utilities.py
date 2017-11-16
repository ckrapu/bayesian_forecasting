from oauth2client.client import GoogleCredentials
import ee
import pandas as pd
import theano.tensor as tt
from theano.compile.ops import as_op
import theano
from scipy.stats import invgamma,genextreme
import matplotlib.pyplot as plt
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display
import numpy as np


def retrieveGridmetSeries(latitude,longitude,bufferInMeters = 5000,
                          seriesToDownload = ['pr','pet','tmmn','tmmx'],
                          startDate = '1979-01-01',endDate   = '2016-12-31',
                          identifier = 'IDAHO_EPSCOR/GRIDMET'):
    credentials = GoogleCredentials.get_application_default()
    ee.Initialize()
    
    point = ee.Geometry.Point(longitude,latitude)

    # Get bounding box
    circle = ee.Feature(point).buffer(bufferInMeters)
    bbox   = circle.bounds()
    
    ic = ee.ImageCollection(identifier)
        
    # Restrict to relevant time period
    ic = ic.filterDate(startDate,endDate)
    ic = ic.select(seriesToDownload)

    # Take average over time across bounding box
    ic =ic.toList(99999).map(lambda x: ee.Image(x).reduceRegion(reducer=ee.Reducer.mean(),geometry=bbox.geometry()))
    
    # Coerce this data into a Pandas dataframe
    df = pd.DataFrame(data=ic.getInfo(),index = pd.date_range(start=startDate,end=endDate)[0:-1])
    return df
 
def retrieveGridmetAtLocations(latitudes,longitudes,returnFrames,**kwargs):
    bigdf = pd.DataFrame()
    frames = []
    for i,latitude in log_progress(enumerate(latitudes),every=1):
    
        try:
            smalldf = retrieveGridmetSeries(latitude,longitudes[i],**kwargs)
            newColumns = ['{0}_{1}'.format(string,i) for string in smalldf.columns]
            smalldf.columns = newColumns
            bigdf = pd.concat([bigdf,smalldf],axis = 1)
            frames.append(smalldf)
        except:
            print 'Retrieval failed for lat/long {0}/{1}'.format(latitude,longitudes[i])
    if returnFrames:
        return frames
    else:
        return bigdf

def tsSamplesPlot(tsSamples,timeIndex = None,upperPercentile = 95, lowerPercentile = 5,ax = None):

    if len(tsSamples.shape) > 2:

        tsSamples = np.squeeze(tsSamples)
        
    if timeIndex is None:
        timeIndex = np.arange(tsSamples.shape[1])
        
    upper  = np.percentile(tsSamples,upperPercentile,axis=0)
    lower  = np.percentile(tsSamples,lowerPercentile,axis=0)
    median = np.percentile(tsSamples,50,axis=0)
    if ax is None:
        plt.plot(timeIndex,upper,linestyle ='--',color='0.4',linewidth = 2)
        plt.plot(timeIndex,lower,linestyle ='--',color='0.4',linewidth = 2)
        plt.fill_between(timeIndex,upper,lower,where=upper>lower,facecolor='0.8',label = '{0} to {1} percentile range'.format(upperPercentile,lowerPercentile))
        plt.plot(timeIndex,median,color='k',linewidth = 3,label = 'Median')
        plt.legend(loc = 'upper right')
        return plt.gca()
    else:
        plt.plot(timeIndex,upper,linestyle ='--',color='0.4',linewidth = 2,axes=ax)
        plt.plot(timeIndex,lower,linestyle ='--',color='0.4',linewidth = 2,axes=ax)
        plt.fill_between(timeIndex,upper,lower,where=upper>lower,facecolor='0.8',label = '{0} to {1} percentile range'.format(upperPercentile,lowerPercentile),axis=ax)
        plt.plot(timeIndex,median,color='k',linewidth = 3,label = 'Median',axes=ax)
    return ax

    
  
    
    
def inverseGammaVisualize(alpha,beta):
    x = np.linspace(invgamma.ppf(0.01, alpha,scale=beta),invgamma.ppf(0.99,alpha,scale=beta), 100)
    plt.plot(x, invgamma.pdf(x, alpha,scale=beta), 'r-', lw=5, alpha=0.6, label='IG density for alpha={0}, beta={1}'.format(alpha,beta))
    plt.legend()
    return plt.gca()
    
       
            
def gev_median(mu,sigma,xi):
    return mu + sigma * (np.log(2.0)**(-xi)-1)/xi
               
 
def pdf_weibull(x,alpha,beta):
    return alpha * x**(alpha-1.0) * np.exp(-(x/beta)**alpha) * (1.0/ beta ** alpha)

    
def log_progress(sequence, every=None, size=None, name='Items'):


    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        ) 
    
    
def observed_vs_medians_gev_plot(trace,observed):
    
    dim0,dim1,dim2 = trace['mus'].shape
    
    sigma_reshaped = trace['sigma'][:,np.newaxis].repeat(dim1,axis=1)[:,:,np.newaxis].repeat(dim2,axis=2)
    xi_reshaped = trace['xi'][:,np.newaxis].repeat(dim1,axis=1)[:,:,np.newaxis].repeat(dim2,axis=2)
    medians = genextreme.median(-xi_reshaped,loc=trace['mus'],scale=sigma_reshaped)
    
    numMedians = len(medians.ravel())
    numObserved = len(observed.values.ravel())
    
    plt.figure(figsize = (8,5))
    sns.kdeplot(medians.ravel(),color='r',label = 'Monte Carlo medians (n = {0})'.format(numMedians))
    sns.kdeplot(observed.values.ravel(),color='b',label='Observed (n = {0})'.format(numObserved))
    plt.ylabel('Density')
    plt.xlabel('Maxima')
    plt.legend(loc='upper right',fontsize = 12)
    return plt.gca()
