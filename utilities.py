import ee
import fiona
import geopandas         as gpd
import itertools
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import shapely
import sys
import theano
import theano.tensor     as tt
import ulmo

from IPython.display               import display
from ipywidgets                    import IntProgress, HTML, VBox
from oauth2client.client           import GoogleCredentials
from pymc3.distributions.dist_math import bound
from scipy.linalg                  import circulant
from scipy.stats                   import invgamma,genextreme
from theano.compile.ops            import as_op

def trace_nonzero(trace,ci=95.0,axis=0):
    assert len(trace.shape) == 2
    medians = np.median(trace,axis = axis)
    upper = np.percentile(trace,100.0-(100.0-ci)/2.0,axis=axis)
    lower = np.percentile(trace,(100.0-ci)/2.0,axis=axis)
    zero_is_between = np.logical_and(upper > 0,lower < 0)
    return ~zero_is_between

def the_only_function_you_need(filepath ='./project/states.shp' ):
    usa     = gpd.read_file(filepath)
    usa     = usa.set_index('STATE_NAME')
    usa     = usa.drop(['Hawaii','Alaska'],axis=0)
    ax      = usa.plot(color='w',figsize = (20,7),edgecolor='k')
    return ax

def gaussianKernel(dist,c):
    return np.exp(-(dist)**2/(2*c**2))

def quadraticKernel(dist,c):
    return 1.0/(c*dist**2)

def euclidDist(pair1,pair2):
    """ Just computes the euclidean distance between
    two tuples, i.e. two latitude/longitude pairs"""
    return ((pair1[0]-pair2[0])**2+(pair1[1]-pair2[1])**2)**0.5

def createK(obsSites,kernelSites,kernel=gaussianKernel,dist=euclidDist,d=4):
    """ Will fill this in later"""
    r = len(obsSites)
    n = len(kernelSites)
    K = np.zeros([r,n])
    for i in range(r):
        for j in range(n):
            K[i,j] = kernel(dist(obsSites[i],kernelSites[j]),d)
    return K



def gev_ll(loc,c,scale):
    """Since PyMC3 has no out-of-the-box solution for GEV-distributed data,
    this function can be used in conjunction with DensityDist to calculate
    the contributions of GEV variables to the log likelihood. The way that
    this works is that when 'gev_ll' is called with some parameters, a 
    function is returned. Credit goes to Adrian Seyboldt
    (https://github.com/aseyboldt) for writing this code."""
    
    def gev_logp(value):
        scaled = (value - loc) / scale
        logp = -(scale
                 + ((c + 1) / c) * tt.log1p(c * scaled)
                 + (1 + c * scaled) ** (-1/c))
        alpha = loc - scale / c
        
        # If the value is greater than zero, then check to see if 
        # it is greater than alpha. Otherwise, check to see if it 
        # is less than alpha.
        bounds = tt.switch(value > 0, value > alpha, value < alpha)
        
        # The returned array will have entries of -inf if the bound
        # is not satisfied. This condition says "if c is zero or
        # value is less than alpha, return -inf and blow up 
        # the log-likelihood.
        return bound(logp, bounds, c != 0)
    return gev_logp

def matrix_normal(M,U,V):
    """ Wrapper for a matrix-variate normal distribution with mean 
    matrix M, column/left covariance matrix U and row/right covariance
    matrix V."""
    return numpy.random.multivariate_normal(M.ravel(), np.kron(V, U)).reshape(M.shape)


def simulate_and_data_matrix_arp(coefficients,sigma = 0.5,length = 100,initial = 1.0,bias = 0.0):
    """Convenience function wrapping together AR(p) simulation
    along with the creation of a regression matrix of lagged values. 
    Note that the coefficient for the farthest-back lag should be placed first
    in the array 'coefficients'."""
       
    # We will make a data vector which is a little too long because
    # some values will need to be thrown out to make the dimension of
    # a lagged data matrix match up with the dimension of y
    p = len(coefficients)
    try:
        assert length > p
    except AssertionError:
        print 'The AR(p) order is larger than the desired time series length.'
        
    y = arp_simulation(coefficients,sigma,length+p,initial = initial,bias = bias)
    
    F = data_matrix_arp_stack(y,p)
    
    # Snip off the values of y for which we cannot assign previous lagged values in the data matrix:
    y = y[p::]
    
    try:
        assert y.shape[0]==F.shape[0]
    except AssertionError:
        print 'The length of the simulated series and data matrices do not agree'
    
    # The different rows of F correspond to different time lags.
    # As the index ranges from 0,...,p, the lag ranges from newest,...,oldest
    return y,F


# TODO: Write unit eigenvalue test to keep bad coefficient values from being set.

def simulate_and_data_matrix_varp(coef_tensor,cov_matrix,length,initial):
    """Convenience function wrapping together VAR(p) simulation
    along with the creation of a regression tensor of lagged values. 
    Note that the coefficient matrix for the most recent lag should be placed first
    in the tensor 'coef_tensor'."""
    
    try:
        p = coef_tensor.shape[2]
    except IndexError:
        p = 1
    
    # We'll make the Y matrix a little longer (in the time dimension)
    # than it needs to be so we can later throw out values which won't
    # have accompanying lagged data in the regression tensor.
    Y = generate_varp(coef_tensor,cov_matrix,length+p,initial)
    F = data_matrix_varp_stack(Y,p)
    
    # The first p entries of Y were only needed to provide lagged data to 
    # prime the following timesteps.
    return Y[p::,:],F
    
def generate_varp(coef_tensor,cov_matrix,length,initial):
    """Use this function to simulate a VAR(p) process over 'length' timesteps
    for an r-dimensional time series with static parameters. 'coef_tensor' must 
    be an array with shape[r,r,p] and 'cov_matrix' must be an array with shape [r,r]
    relating the innovations across different entries in the vector time series.
    Also, the coefficient array passed to this function should have the most recent
    lag coefficients come before the coefficients for lag values in the more distant past.
    In practice, this means that the slice 'coef_tensor[:,:,0]' corresponds to the p = 1 lag
    while the slice 'coef_tensor[:,:,10]' corresponds to the p = 10 lag.
    
    The ordering of the dimensions for coef_tensor should look like:
    [out_index,in_index,time] where out_index denotes that this index runs over regression outputs
    while in_index runs over regression inputs. Therefore, an index of 2,3,4 denotes
    the regression coefficient relating the contribution of the 3rd series to the 2nd series at 
    time lag 4."""
    
    # We first want to make sure the coefficient matrix we've received is 
    # square in its first two dimensions.
    try:
        assert coef_tensor.shape[0]==coef_tensor.shape[1]
    except AssertionError:
        print 'Coefficient tensor is not square in first two dimensions.'
        
    try:
        p = coef_tensor.shape[2]
    except IndexError:
        p = 1
    
    r = coef_tensor.shape[0]
    
    y = np.empty([length,r])
    
    # We'll sample all our errors at once. Innovations should be of dimension length x r
    innovations = np.random.multivariate_normal(np.zeros(r),cov_matrix,size = length)
    
    for t in range(length):
        
        # If the timestep is less than the VAR(p) order, then there won't be enough
        # previous data to build a lagged data set
        if t < p:
            y[t,:] = initial
        else:
            # We snap off a block of recent values
            # with shape [r,p].
            recent_y_matrix = y[t-p:t,:]
            
            # Since the time index runs like low:high implies past:recent,
            # we need to invert the time axis because the coefficient tensor
            # has a time index running like low:high implies recent:past,
            # i.e. p=1 comes first
            reversed_recent = np.flipud(recent_y_matrix)
                     
            # Then, we use einsum to perform a tensor contraction 
            # and then we add the innovations.
            y[t,:] = np.einsum('ikj,jk->i',coef_tensor,reversed_recent) + innovations[t,:]
            
    return y

def data_matrix_varp_stack(Y,p):
    """ This function takes in an r-dimensional time series 'y' and
    rearranges/repeats the data to create a regression tensor of 
    lag 1, lag 2, ... lag 'order' values for use in an autoregression
    fitting procedure."""
    
    r = Y.shape[1]
    T = Y.shape[0]
    F = np.zeros([T,p,r])
    for i in range(p):
        # We'll add a slice of data lagged by 'i' timesteps.
        F[:,i,:] = np.roll(Y,i+1,axis = 0) 
    return F[p::,::-1,:]
    
    


def data_matrix_arp_stack(y,p):
    """ This function takes in a 1-dimensional time series 'y' and
    rearranges/repeats the data to create a regression matrix of 
    lag 1, lag 2, ... lag 'order' values for use in an autoregression
    fitting procedure. More memory efficient than data_matrix_arp_circulant."""
    
    T = len(y)
    F = np.zeros([T,p])
    for i in range(p):
        F[:,i] = np.roll(y,i+1,axis = 0)[:,0]
    
    return F[p::,:]

def data_matrix_arp_circulant(y,p):
    """ This function takes in a 1-dimensional time series 'y' and
    rearranges/repeats the data to create a regression matrix of 
    lag 1, lag 2, ... lag 'order' values for use in an autoregression
    fitting procedure. This function is conceptually cleaner but much 
    more memory intensive than data_matrix_arp_stack."""
    
    circ = circulant(y).T
    return circ[0:p,p-1:-1].T

def arp_simulation(coefficients,sigma,length,initial = 1.0,bias = 0.0):
    """Simulate a 1-dimensional AR(p) process of duration 'length'
    given initial starting value of 'initial'. The standard deviation
    of the error term is given by 'sigma'.
    The coefficient array passed to this function should
    be arranged from lowest order to highest. For example,
    the 3rd order coefficient in a simulated AR(3) process
    would be passed in as the first coefficient in the 
    coefficient array. 'bias' controls the mean of the innovation."""
    
    p = len(coefficients)
    
    # Here, we reverse the view to jive with the indexing later on.
    coefficients = coefficients[::-1]
    
    # We will make a data vector which is a little too long because
    # some values will need to be thrown out to make the dimension of
    # a lagged data matrix match up with the dimension of y
    y = np.zeros([length,1]) # This is arranged as a length x 1 vector for easier handling down the line.
    y[0:p] = initial
    innovations = np.random.normal(loc = bias,scale = sigma,size = [length,1])
    for i in range(p,length):
        y[i] = coefficients.dot(y[i-p:i]) + innovations[i]
    return y

def univariate_dlm_simulation(F,G,W,v,initial_state,n,T):
    """This function is used to simulate a univariate DLM with static
    parameters F,G,W,v. The initial state for the state vector is 
    specified with 'initial_state'. 'n' and 'T' are the 
    dimensions of the state vector and the number of timesteps
    desired, respectively. """
    
    ZEROS = np.zeros(n)
    
    emissions    = np.zeros([T,1])
    state        = np.zeros([T,n])
    
    state[0]     = initial_state
    emissions[0] = F.dot(initial_state) + np.random.normal(loc = 0.0,scale = v)
    
    for t in range(T):
        state[t] = G.dot(state[t-1]) + np.random.multivariate_normal(ZEROS,W)
        emissions[t] = F.dot(state[t]) + np.random.normal(0.0, v)
        
    return state,emissions
        
    
def permutation_matrix(order):
    """Produces a permutation matrix
    with dimension 'order'."""
    matrix = np.zeros([order,order])
    matrix[-1,0] = 1
    matrix[0:-1,1::] = np.identity(order-1)
    return matrix

def polynomial_matrix(order):
    """Produces a matrix useful for including
    a polynomial component of order 'order' in G 
    for a DLM."""
    
    matrix = np.identity(order)
    for i in range(order-1):
        matrix[i,i+1] = 1
    return matrix
    
def water_year_means(df):
    """This function takes in a daily timestep indexed dataframe 'df'
    and takes the group mean with the groups defined by the water year
    of the observation. For example, the water year for 2002 is given 
    by October 1, 2001 through September 31, 2002."""

    monthly = df.groupby(pd.TimeGrouper('M')).mean()
    monthly['year'] = monthly.index.year
    monthly['month'] = monthly.index.month
    monthly['water_year'] = np.roll(monthly['year'],-3)
    
    # Because there will typically not be data starting and ending in
    # October, we will need to drop the first and last years as we have
    # incomplete records for the first and last year respectively.
    annual = monthly.groupby(monthly['water_year']).mean().iloc[1:-1]
    return annual.drop(['year','month'],axis=1).set_index('water_year')

def parse_mopex(filename):
    """ Since the MOPEX hydrology dataset uses a fixed-character field scheme,
    this function just takes care of parsing it, replacing the -99999 
    values with NaNs and setting the correct time index."""
    
    columnNames = ['date','precipitation','pet','discharge','max_temp','min_temp']

    data = pd.read_csv(filename,sep=r"[ ]{2,}",names=columnNames)
    data['year'] = data['date'].apply(lambda x: x[0:4])
    data['month'] = data['date'].apply(lambda x: x[4:6])
    data['day'] = data['date'].apply(lambda x: x[6:8])
    data = data.set_index(pd.to_datetime(data[['year','month','day']]))
    data = data.replace(to_replace=-99.0000,value=np.nan)
    return data.drop(['date','year','month','day'],axis = 1)

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

def tsSamplesPlot(tsSamples,timeIndex = None,upperPercentile = 95, lowerPercentile = 5,ax = None,color ='k',median_label = 'Median',fill_label = '90% CI',fill_alpha = 0.25):

    if len(tsSamples.shape) > 2:
        tsSamples = np.squeeze(tsSamples)
        
    if timeIndex is None:
        timeIndex = np.arange(tsSamples.shape[1])
        
    upper  = np.percentile(tsSamples,upperPercentile,axis=0)
    lower  = np.percentile(tsSamples,lowerPercentile,axis=0)
    median = np.percentile(tsSamples,50,axis=0)
    if ax is None:
        plt.plot(timeIndex,upper,linestyle ='--',color=color,linewidth = 2)
        plt.plot(timeIndex,lower,linestyle ='--',color=color,linewidth = 2)
        plt.fill_between(timeIndex,upper,lower,
                         where=upper>lower,facecolor=color,alpha = fill_alpha,label = fill_label)
        plt.plot(timeIndex,median,color=color,linewidth = 3,label = median_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
        return plt.gca()
    else:
        plt.plot(timeIndex,upper,linestyle ='--',color=color,linewidth = 2,axes=ax)
        plt.plot(timeIndex,lower,linestyle ='--',color=color,linewidth = 2,axes=ax)
        plt.fill_between(timeIndex,upper,lower,
                         where=upper>lower,facecolor=color,alpha = fill_alpha,label = fill_label,axis=ax)
        plt.plot(timeIndex,median,color=color,linewidth = 3,label = median_label,axes=ax)
    return ax

def get_ghcn_data(upper_latitude,lower_latitude,left_longitude,right_longitude,country=None,element = 'TMAX',**kwargs):
    """ This is a wrapper designed to make retreiving daily data from the GHCN database 
    more streamlined. Specify 'element' as one of PRCP, TMAX, TMIN, SNOW or SNWD. Other codes 
    might be retrievable too - see ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt for 
    more details. If locations within the specified coordinate box are desired only from a 
    specific country, then use the keyword argument 'country' to restrict to those locations. 
    This function returns a dataframe in which each column corresponds to the desired weather 
    variable for a single station and the index runs over daily timesteps.
    """
    
    assert lower_latitude < upper_latitude
    assert left_longitude < right_longitude
    
    # Because this API is unpatriotic and doesn't love the full name of our wonderful homeland.
    if country == 'USA':
        country = 'US'
    
    stations = pd.DataFrame(ulmo.ncdc.ghcn_daily.get_stations(country=country,elements=element,**kwargs))
    
    stations = stations.transpose()
    sys.stdout.flush()
    subset_stations = stations[stations.latitude.between(lower_latitude,upper_latitude) & stations.longitude.between(left_longitude,right_longitude)]
    print 'Retrieving {0} data for {1} stations.'.format(element,subset_stations.shape[1])
    frames = []
    
    for station_id in log_progress(subset_stations.index,every=1):
        frame = ulmo.ncdc.ghcn_daily.get_data(station_id,elements = element,as_dataframe=True)[element][['value']]
        frame.columns = [station_id]
        frames.append(frame)
    
    merged = pd.concat(frames,axis = 1)
    
    return merged,subset_stations   

def grid_in_shape(up_lat,low_lat,left_long,right_long,shape,
                  lat_resolution=15,long_resolution=60):
    """This function takes the bounding box of a grid with
    specified longitudinal and latitudinal resolution 
    along with a shapely shape and returns the grid points
    that are located within that shape, as a geodataframe.""" 
      
    longitudes = np.linspace(left_long,right_long,60)
    latitudes  = np.linspace(low_lat,up_lat,15)
    prods = list(itertools.product(longitudes,latitudes))
    points = [shapely.geometry.Point(point) for point in prods]
    points_within = [point for point in points if shape.contains(point)]
    points_gdf = gpd.GeoDataFrame(geometry = points_within)
    
    return points_gdf
  
def multi_impute_maxes(frame,daily_limit = 15,annual_limit = 1,
                max_nan_per_year = 60):
    """ This function combines imputation at the daily 
    and annual timescale for a pandas dataframe indexed
    by a daily timestamp. First, it makes a pass over the 
    data and fills gaps of length up to 2 x 'daily_limit'
    by equal amounts of backward and forward fill. Then,
    If there are more than 'max_nan_per_year NaNs remaining 
    in a year,that year is flagged as an NaN year and 
    an additional step of forward and backward filling
    is applied to fill gaps of length 2 x 'annual_limit.
    Any columns which have remaining NaN values are 
    dropped from the returned matrix."""
    
    n_columns_original = frame.shape[1]
    
    # We first do a backward/forward fill to cover gaps up 
    # to 2*daily_limit days.
    frame_bfill = frame.fillna(method='bfill',limit = daily_limit)
    frame_ffill = frame_bfill.fillna(method='ffill',limit = daily_limit)

    # We then compute a semi-processed maxima dataframe. 
    frame_max   = frame_ffill.groupby(pd.TimeGrouper('A')).max()

    # Furthermore, we want to NaN out any years for which there are 
    # too many unobserved days.
    nans_per_year = np.isnan(frame).groupby(pd.TimeGrouper('A')).sum()

    # If there are too many NaNs in that year, we drop that year.
    max_nan_per_year = 60
    is_year_allowed = nans_per_year < max_nan_per_year # Boolean array
    frame_max[~is_year_allowed] = np.nan

    # Next, we back/forward fill 'annual_limit' years. 
    # Any stations which still have missing data 
    # remaining are dropped from further analysis.
    max_bfill    = frame_max.fillna(method='bfill',limit = annual_limit)
    max_ffill    = max_bfill.fillna(method='ffill',limit = annual_limit)
    max_dropped  = max_ffill.dropna(axis=1)
    
    n_dropped = max_dropped.shape[1] - n_columns_original
    print 'Out of {0} columns, {1} were dropped.'.format(n_columns_original,
                                                        n_dropped)
    
    return max_dropped
    
    
def inverseGammaVisualize(alpha,beta):
    x = np.linspace(invgamma.ppf(0.01, alpha,scale=beta),invgamma.ppf(0.99,alpha,scale=beta), 100)
    plt.plot(x, invgamma.pdf(x, alpha,scale=beta),
             'r-', lw=5, alpha=0.6, label='IG density for alpha={0}, beta={1}'.format(alpha,beta))
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
 