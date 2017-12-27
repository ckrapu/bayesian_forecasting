import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import r2_score
from scipy.stats import norm
from scipy.stats import t as student_t

class FFBS_sample(object):
    """An FFBS_sample object is used as a sampling step in a PyMC3 model 
    in order to draw samples of the DLM state vector conditional on 
    emissions/observations and the various DLM structures F,G,W,V. The dict
    'varnames_mapping' links the names of the relevant DLM quantities in the
    larger PyMC3 model to their desired position in the FFBS data structure."""

    # The FFBS_sample object is only initialized once in every MCMC run.
    # Therefore, the only arguments that the constructor takes should be 
    # variables which DO NOT change across Monte Carlo samples. This is 
    # why the observations Y are not passed in here - they can change from
    # MC iteration to iteration and thus need to be specified in the body
    # of the 'step' method.
    def __init__(self,vars,F,G,
                 varnames_mapping={'m0':'m0',
                                   'C0':'C0',
                                   's0':'s0',
                                   'state':'state',
                                   'V':'V',
                                   'W':'W',
                                   'Y':'Y'},
                 exponentiate_W = False,exponentiate_V = False,evo_discount_factor = [0.99],obs_discount_factor = 0.99,calculate_ll = False):
               
        self.F = F
        self.G = G
        
        # The attribute 'vars' must be set for PyMC3 to 
        # handle this step object appropriately.
        self.vars             = vars
        self.varnames_mapping = varnames_mapping
        
        # Calculating the log-likelihood is not necessary to draw a MC sample and
        # it might be really slow. Usually, we want to bypass this.
        self.calculate_ll     = calculate_ll
        
        self.evo_discount_factor = evo_discount_factor
        self.obs_discount_factor = obs_discount_factor
        
        # Since PyMC3 usually works with the logarithm of positive random variables,
        # it is handy to be able to tell the FFBS sampler to exponentiate the log-variances
        # and recover the actual variances.
        self.exponentiate_W = exponentiate_W
        self.exponentiate_V = exponentiate_V
        
                

    def step(self,estimate):
        
        F = self.F
        G = self.G
         
        # We don't need to specify the evolution and observation variances
        # and if we don't, then we'll just use a discount approach later.
        try:
            W_varname    = self.varnames_mapping['W']
            W            = self.vars[W_varname]
            evo_discount = False
            assert np.all(W > 0)
            
        except KeyError:
            W  = None
            evolution_discount = True
            
        try: 
            V_varname    = self.varnames_mapping['V']
            V            = self.vars[V_varname]
            obs_discount = False
            assert np.all(V > 0)
            
        except KeyError:
            V  = None
            obs_discount = True
        
        
        # The prior mean, state covariance
        # and observations always need to be specified.
        m0_varname = self.varnames_mapping['m0']
        m0         = estimate[m0_varname]
        
        C0_varname = self.varnames_mapping['C0']
        C0         = estimate[C0_varname]
      
        Y_varname = self.varnames_mapping['Y']
        Y         = estimate[self.Y_varname]
        
        # If there is no observational variance specified, then
        # we need to get the prior on the observational variance.
        if V is None:
            s0_varname = self.varnames_mapping['s0']
            s0         = estimate[s0_varname]

        self.ffbs = ffbs(F,G,Y,m0,C0,W=W,V=V,s0=s0,
                         obs_discount = obs_discount,
                         evolution_discount = evolution_discount,
                         obs_discount_factor = self.obs_discount_factor,
                         evo_discount_factor = self.evo_discount_factor,
                         calculate_ll = self.calculate_ll)
        
        self.ffbs.forward_filter()
        self.ffbs.backward_smooth()
        
        new_state = self.ffbs.backward_sample()
        new_estimate = estimate.copy()
        
        state_varname = varnames_mapping['state']
        new_estimate[state_varname] = new_state

        return new_estimate

class FFBS(object):
    def __init__(self,F,G,Y,m0,C0,nancheck = True,check_cov_pd=False,
                 evolution_discount=True,obs_discount = True,
                 evo_discount_factor = [0.98],obs_discount_factor = 0.98,
                 W=None,V=None,s0 = 1.0,warn_G_singular = False,dynamic_G = False,
                 calculate_ll = True,model_id = None):
        
        # The convention we are using is that F  must be specified as a sequence
        # of [n,r] arrays respectively.
        # The first dimension runs over time and so we must make sure that the
        # arrays fed in have 3 dimensions. Otherwise, we'll run into problems
        # down the road.
        assert len(F.shape) == 3
        
        if dynamic_G:
            # It is useful to be able to specify a G matrix that changes.
            # This is especially helpful for seasonal effects with a long
            # period.
            assert len(G.shape) == 3
            assert Y.shape[0]   == G.shape[0] # Check to see if number of observations matches number of G
            assert G.shape[1]   == G.shape[2] # Check to see if square
            assert G.shape[1]   == F.shape[1] # Make sure column index is shared between F,G
        
        else:
            # This is the case where we want a single G matrix for all time.
            assert len(G.shape) == 2
         
        self.check_cov_pd         = check_cov_pd          # Do we want to warn if covariance is not positive definite?
        self.evolution_discount   = evolution_discount    # Discount factor for state evolution variance
        self.obs_discount         = obs_discount          # Is the observational variance observed?
        self.warn_G_singular      = warn_G_singular       # Warning for G not being singular
        self.dynamic_G            = dynamic_G             # Is G allowed to vary over time?
        self.obs_discount_factor  = obs_discount_factor   # Discount factor for observational variance
        self.evo_discount_factor  = evo_discount_factor   # Discount factor for system evolution variance
        self.s0                   = s0                    # Prior point estimate of observational variance
        self.calculate_ll         = calculate_ll          # Allows us to turn off likelihood calculations if too slow
        self.model_id             = model_id              # Set this to anything you want
        self.is_filtered          = False
        self.is_backward_smoothed = False
        self.nancheck             = nancheck              # Determines whether we check to see if NaNs have appeared
 
        # The dimension of the observed series needs to be determined here.
        # Most of the code will only work for the univariate case; the 
        # multivariate version isn't operational yet.
        try:
            obs_dim = Y.shape[1]
        except IndexError:
            obs_dim= 1
               
        self.T         = Y.shape[0]   # Number of timesteps
        self.obs_dim   = obs_dim            # Observation dimension
        self.state_dim = m0.shape[0]  # State dimension

        # F and G always need to be specified.
        self.F    = F    # Dynamic regression vectors
        self.G    = G    # Evolution matrix
        
        # If the matrix V is not specified (and it usually isn't) the the observational
        # variance is treated as an unknown variable.
        if self.obs_discount: 
            assert V is None
            self.s0 = s0
                 
        else:
            self.V    = V    # Static observation variance with dimension [r,r]
            
        self.Y    = Y    # Observations with dimension [T,r]
        self.m0   = m0   # Prior mean on state vector with dimension [n,1]
        self.C0   = C0   # Prior covariance on state vector with dimensions [n,n]
            
        # We need to make sure that the DLM evolution variance is specified one way or another.
        if evolution_discount:
            
            # For retrospective analysis, G needs to be nonsingular with a 
            # discount approach.
            if warn_G_singular:
                try:
                    np.linalg.cholesky(G) # Fails if G is singular.
                except np.linalg.LinAlgError:
                    print 'A discount factor was specified but G is singular. Retrospective analysis will not be reliable.'
        else:
            # Fire off an error if the evolution variance is not specified one way or another.
            if W is None:
                raise ValueError('Neither a discount factor nor evolution variance matrix has been specified.')
            else:
                self.W    = W    # Static state evolution variance matrix
 
        # The discount factors should be passed in as a list of delta values
        # with one delta for each dimension of the state vector.
        # If there is only one delta passed in, then we'll use it as the 
        # global discount factor.
        if len(evo_discount_factor) == 1:
            self.discount_matrix = np.identity(self.state_dim) *(1.0 /  evo_discount_factor[0])

        elif len(evo_discount_factor) == self.state_dim:
            # The diagonal entries of the discount matrix need to be 
            # 1/delta. We just invert a diagonal matrix to get that.
            self.discount_matrix = np.linalg.inv(np.diag(evo_discount_factor))
            
        # If neither 1 nor n discount factors was passed, then it's unclear
        # exactly which discount factors align with which state elements.
        # Later on, this could be reworked to allow for block discounting.
        else:
            raise ValueError('Evolution discount factors incorrectly specified.')

    def forward_filter(self):

        T         = self.T         # Number of timesteps
        obs_dim   = self.obs_dim   # Dimension of observed data
        state_dim = self.state_dim # Dimension of state vector
        
        if self.obs_discount:
            self.gamma_n = np.zeros(T)
            self.s       = np.zeros(T)
            self.s[0]    = self.s0
           
        else:
            V = self.V # Dimensions of [obs_dim,obs_dim]
            
        self.r = np.zeros(T)       # For unknown obs. variance
        self.e = np.zeros([T,obs_dim])   # Forecast error
        self.f = np.zeros([T,obs_dim])   # Forecast mean
        self.m = np.zeros([T,state_dim])   # State vector/matrix posterior mean
        self.a = np.zeros([T,state_dim])   # State vector/matrix prior mean
        self.Q = np.zeros([T,obs_dim,obs_dim]) # Forecast covariance
        self.A = np.zeros([T,state_dim,obs_dim]) # Adaptive coefficient vector
        self.R = np.zeros([T,state_dim,state_dim]) # State vector prior variance
        self.C = np.zeros([T,state_dim,state_dim]) # State vector posterior variance
        self.B = np.zeros([T,state_dim,state_dim]) # Retrospective ???
        
        # If we want to change the tracked quantities all at once later,
        # it would be handy to be able to reference all of them at the 
        # same time.
        self.dynamic_names = ['F','Y','r' , 'e', 'f' ,'m' ,'a', 'Q', 'A', 'R','C','B']
        
        if self.obs_discount:
            self.dynamic_names = self.dynamic_names + ['gamma_n','s']
        if self.dynamic_G:
            self.dynamic_names = self.dynamic_names + ['G']

        # Forward filtering
        # For each time step, we ingest a new observation and update our priors
        # to posteriors.
        for t in range(T):
            self.t = t
            self.filter_step(t)
                        
        # The last thing we want to do is tabulate the current
        # step's contribution to the overall log-likelihood.
        if self.calculate_ll:
            if self.obs_discount:
                # We need the shape parameters for the preceding time step in the current
                # timestep's calculation of the log likelihood. This just offsets the 
                # vector of shape parameters.
                shifted_gamma = np.roll(np.squeeze(self.gamma_n),1)
                shifted_gamma[0]  = 1.0
                self.log_likelihood = student_t.logpdf(np.squeeze(self.e),
                                                          shifted_gamma,
                                                          scale=np.squeeze(np.sqrt(self.Q)))
            else:
                self.log_likelihood = norm.logpdf(np.squeeze(self.e), 
                                                     scale=np.squeeze(np.sqrt(self.Q)))

            # This is the marginal model likelihood.
            self.ll_sum = np.sum(self.log_likelihood)
        
        if self.nancheck:
            try:
                for array in [self.A,self.C,self.Q,self.m,self.log_likelihood]:
                    assert np.any(np.isnan(array)) == False
                          
            except AssertionError:
                print 'NaN values encountered in forward filtering.'
        
        self.populate_scores()
        
        self.is_filtered = True
        
    def populate_scores(self):
        self.mae = np.mean(np.abs(self.e))
        self.mse = np.mean((self.e)**2)
        self.r2 = r2_score(self.Y,self.f)
        
    def filter_step(self,t):
        # If G varies over time, pick out the one we want.
        if self.dynamic_G:
            G = self.G[t]
        else:
            G = self.G # Dimensions of [n,n]

        # If starting out, we use our initial prior mean and prior covariance.
        prior_covariance = self.C0 if t == 0 else self.C[t-1]
        prior_mean       = self.m0 if t == 0 else self.m[t-1]
        self.a[t]        = G.dot(prior_mean)
        
        # If we use a discounting approach for the evolution variance,
        # we just inflate our covariance matrix by a little bit 
        # at each time step. Otherwise, we just add the innovation
        # variance W. We do this because the covariance of the sum of 
        # two normal RVs is equal to the elementwise sum of the covariances.
        if self.evolution_discount:
            self.R[t]   = G.dot(prior_covariance).dot(G.T).dot(self.discount_matrix)
        else:
            self.R[t]   = G.dot(prior_covariance).dot(G.T) + self.W

        # The one-step forecast 'f' involves the product of our regression 
        # matrix F and the state vector.
        self.f[t]   = self.F[t].T.dot(self.a[t])
        self.e[t]   = self.Y[t] - self.f[t]
        
        # Next, we calculate the forecast covariance and forecast error.
        # If we don't know the observational variance, then we need
        # to keep track of the prior-to-posterior updating of our distribution
        # over the observational variance.
        if self.obs_discount:
            
            if t == 0:
                self.Q[t]       = self.F[t].T.dot(self.R[t]).dot(self.F[t]) + self.s0
                self.gamma_n[t] = 1.0
                self.r[t]       = (self.gamma_n[t] + self.e[t]**2 / self.Q[t]) / (self.gamma_n[t] + 1)
             
            else:
                self.Q[t]       = self.F[t].T.dot(self.R[t]).dot(self.F[t]) + self.s[t-1]
                self.gamma_n[t] = self.obs_discount_factor * self.gamma_n[t-1]+1
                self.r[t]       = (self.gamma_n[t] + self.e[t]**2 / self.Q[t]) / (self.gamma_n[t] + 1)
                self.s[t]       = self.r[t] * self.s[t-1]
        
        # In the case where the observational variance is known, the forecast variance
        # is expressed much more succinctly.
        else:
            self.Q[t]   = self.F[t].T.dot(self.R[t]).dot(self.F[t]) + self.V #
            
        # The ratio of R / Q gives us an estimate of the split between
        # prior covariance and forecast covariance.
        if self.obs_discount and t > 0 :
            self.prefactor = self.s[t]/self.s[t-1]
        else:
            self.prefactor = 1.0
        if self.obs_dim == 1:
            self.A[t] = self.R[t].dot(self.F[t])/np.squeeze(self.Q[t])
            self.C[t] = self.prefactor * (self.R[t] - self.A[t].dot(self.A[t].T)*np.squeeze(self.Q[t]))
        else: # This branch is probably broken as I have not tested any of it for a mv case.
            self.A[t] = self.R[t].dot(self.F[t]).dot(np.linalg.inv(self.Q[t]))
            self.C[t] = self.prefactor * (self.R[t] - self.A[t].dot(self.Q[t]).dot(self.A[t].T))
        
        # The posterior mean over the state vector is a weighted average 
        # of the prior and the error, weighted by the adaptive coefficient.            
        self.m[t,:]   = self.a[t]+self.A[t].dot(self.e[t])
        
    def append_observation(self,new_F, new_y):
        
        assert self.is_filtered
        
        self.T = self.T + 1
        
        # We need to extend all of our arrays to hold the new computed values.
        # The first axis should always be over timesteps, so we append
        # our new data in that axis.
        for array_name in self.dynamic_names:
            
            old_array   = getattr(self,array_name)
            array_shape = old_array.shape
            
            # The next line makes sure our  piece to add on has the same number of
            # dimensions as the base part. Otherwise, concatenate will hit an error.
            addendum    = np.zeros([1] + list(array_shape[1::]))
            setattr(self,array_name,np.append(old_array, addendum, axis = 0))
        
        self.F[-1] = new_F
        self.Y[-1] = new_y
        
        # Last, we want to do filtering on the final timestep which we just added.
        self.filter_step(self.T - 1)
        
        # And we will also update the error metrics.
        # TODO: rewrite populate_scores so that a full recompute is not applied everytime
        self.populate_scores()
        
    def backward_smooth(self):
        """ This method is used to compute retrospective estimates
        of the DLM state vector after we have made a first pass over the
        data with the forward_filtering method."""
        
    
        # None of the necessary estimates required for the BS step will be ready
        # if we haven't already applied the forward filtering.
        try:
            assert self.is_filtered
        except SystemError:
            print('The forward filtering process has not been applied yet.')

        # These are the main quantities we care about.
        # The suffix _r indicates that these are from retrospective analyses.
        # s_r and n_r are used only if the observational variance is constant but unknown.   
        self.m_r =  np.zeros([self.T,self.state_dim])         # Retrospective mean over state distribution
        self.C_r =  np.zeros([self.T,self.state_dim,self.state_dim])  # Retrospective posterior covariance over state
        self.s_r =  np.zeros(self.T)                  # Retrospective smoothed estimate of observational variance
        self.n_r =  np.zeros(self.T)                  # Retrospective smoothed degrees of freedom
        
        # The loop runs from the final time step to the first.
        for t in range(self.T-1,-1,-1):
                # Unlike in the case of an unknown evolution matrix with discounting,
                # we don't need the inverse of G or G transpose.
                if self.dynamic_G:
                    G = self.G[t+1]
                                      
                else:
                    G = self.G 
                    
                if t == ( self.T-1):
                    self.m_r[t] = self.m[t] 
                    self.C_r[t] = self.C[t]
                    
                    if self.obs_discount:
                        # Set smoothed estimate of observational variance to be 
                        # the last forward-filtered estimate of variance
                        self.s_r[t] = self.s[t]
                        self.n_r[t] = self.gamma_n[t]
                    
                else:
                    self.B[t]   = self.C[t].dot(G.T).dot(np.linalg.inv(self.R[t+1]))
                    self.m_r[t] = self.m[t] + self.B[t].dot(self.m_r[t+1] -  self.a[t+1])
                    self.C_r[t] = self.C[t] + self.B[t].dot(self.C_r[t+1] -  self.R[t+1]).dot(self.B[t].T)
                    if self.obs_discount:
                        self.s_r[t] = ((1.0 - self.obs_discount_factor) / self.s[t] + self.obs_discount_factor / self.s_r[t+1])**-1
                        self.n_r[t] = (1-self.obs_discount_factor) * self.gamma_n[t] + self.obs_discount_factor * self.gamma_n[t+1]
                        self.C_r[t] = self.C_r[t] * self.s_r[t]/self.s[t]
                                    
        if self.nancheck:
            for array in [self.m_r,self.C_r,self.B]:
                try:
                    assert np.any(np.isnan(array)) == False

                except:
                    print 'NaN values encountered in backward smoothing.'

    def backward_sample(self,num_samples = 1):
        """ This method is used to sample a trajectory of possible DLM states
        through time. It can only be used once the object has gone through filtering
        and backward smoothing."""
        
        T = self.T
        self.theta = np.zeros([T,self.state_dim,num_samples])
        
        # TODO: optimize code to vectorize the sample drawing
        for i in range(num_samples):
            self.sample_C = np.zeros([T,self.state_dim,self.state_dim])
            self.sample_m = np.zeros([T,self.state_dim])

            # Depending on whether the observational variance is known or 
            # not, we proceed using one of two different recursive sampling algorithms.
            if self.obs_discount:

                # First, we draw a sample variance by making a gamma draw from the 
                # posterior over the precision and inverting it.
                self.sample_precision = np.random.gamma(self.gamma_n[-1]/2.0,scale = 2.0 / (self.gamma_n[-1]*self.s[-1]))
                
                # TODO: implement backward sampling of variances
                self.sample_v = 1.0 / self.sample_precision

                # At the beginning (aka the last time step) we assume that the prior
                # over the sample theta is given by our posterior over the state vector
                # at the last time step.
                self.sample_m[-1]  = self.m[-1]
                self.sample_C[-1]  = self.sample_v *self.C[-1]/self.s[-1]
                self.theta[-1,:,i] = np.random.multivariate_normal(self.sample_m[-1], self.sample_C[-1])
 
                # We iteratively draw a sample of theta, condition on that sample and draw
                # a new value of theta for the preceding timestep.
                for t in range(T-2,-1,-1):
                    self.t = t
                    self.sample_B     = self.C[t].dot(self.G.T).dot(np.linalg.inv(self.R[t+1]))
                    self.sample_C[t]  = self.sample_v * (self.C[t] - self.sample_B.dot(self.R[t+1]).dot(self.sample_B.T)) / self.s[t]
                    self.sample_m[t]  = self.m[t] + self.sample_B.dot(self.theta[t+1,:,i] - self.a[t+1])
                    self.theta[t,:,i] = np.random.multivariate_normal(self.sample_m[t], self.sample_C[t])

            else:
                # The backward sampling is much simpler if the observational variance is known 
                # and the code below uses a known observational variance.
                # see page 130 in Prado & West for the details.
                self.sample_m[-1]  = self.m[-1]
                self.sample_C[-1]  = self.C[-1]
                self.theta[-1,:,i] = np.random.multivariate_normal(self.sample_m[-1], self.sample_C[-1])
                for t in range(T-2,-1,-1):
                    self.t = t
                    self.sample_m[t] = self.m[t+1] + self.B[t].dot(self.theta[t+1,:,i] - self.a[t+1])
                    self.sample_C[t] = self.C[t] - self.B[t].dot(self.R[t+1]).dot(self.B[t].T)
                    self.theta[t,:,i] = np.random.multivariate_normal(self.sample_m[t],self.sample_C[t])


        if self.nancheck:
            assert ~np.any(np.isnan(self.theta))
                
        return self.theta
    
    
    def pred_vs_obs_plot(self,figsize =(6,6) ):
        """ Wrapper for scatter plot showing the predicted
        versus observed values"""
        
        # There won't be anything to plot if we haven't done the forward filtering.
        assert self.is_filtered
        
        # This automatically determines the figure limits
        low = min([np.min(self.f),np.min(self.Y)]) * 0.9
        high = max([np.max(self.f),np.max(self.Y)])* 1.1
        
        plt.figure(figsize = figsize )
        plt.scatter(self.Y,self.f,color='k')
        plt.ylabel('Predicted',fontsize = 14)
        plt.xlabel('Observed',fontsize = 14)
        plt.xlim([low,high])
        plt.ylim([low,high])
        plt.plot([low,high],[low,high],linestyle='--',color = 'k')
        
        # Equal axis units are very helpful.
        plt.gca().set_aspect('equal', adjustable='box')
        
        return plt.gca()
    
    def time_plot(self,deviation_multiplier = 1.645,figsize = (10,6),start_index=1,stop_index = -1):
        """Wrapper for a plot showing the forecasted values with 90% 
        confidence interval and the observed values. The first timestep
        is ignored in order to avoid distortion from misspecified 
        variance."""
        
        deviations = np.squeeze(np.sqrt(self.Q[start_index:stop_index]) * deviation_multiplier)
        upper = np.squeeze(self.f[start_index:stop_index])+deviations
        lower = np.squeeze(self.f[start_index:stop_index])-deviations
        
        plt.figure(figsize = figsize)
        plt.plot(self.f[start_index:stop_index],linestyle='-',color='k',label='1-step forecast')
        plt.plot(self.Y[start_index:stop_index],color='r',linestyle='',marker='o',label='Observed')
        plt.gca().fill_between(np.arange(len(deviations)),upper,lower,color='0.8',label='90% CI')
        
        plt.legend(loc='upper right')
        return plt.gca()

    def residual_plot(self,figsize = (6,6)):
        """Wrapper for plot comparing the residuals / forecast errors
        against the observations. A horizontal line is added at residual = 0
        in order to aid visual identification of heteroscedasticity."""
        
        plt.figure(figsize = figsize)
        low  = np.min(self.Y) * 0.9
        high = np.max(self.Y)* 1.1

        plt.scatter(self.Y,self.e,color = 'k')
        plt.ylabel('Residual',fontsize = 14)
        plt.xlabel('Observed',fontsize = 14)
        plt.plot([low,high],[0,0],linestyle='--',color = 'k')
       
        return plt.gca()

        
class GridSearchDiscountFFBS(object):
    """This class is designed to simplify the selection of discount factors
    in the case of unknown observational and evolution variance for the DLM.
    Currently, only an exhaustive grid search is allowed."""
    # TODO: implement latin hypercube sampling and allow for distinct block discount factors
    # TODO: break this into an initialization and an optimization method
    
    def __init__(self, evo_discount_range, obs_discount_range,
                 F,G,Y,m0,C0,s0 = 1.0):
        
               
        self.evo_list = list(evo_discount_range)
        self.obs_list = list(obs_discount_range)
        
         
        # The product iterator needs to be explictly enumerated. 
        # the call to list() forces the evaluation.s
        self.combinations = list(itertools.product(evo_discount_range,obs_discount_range))
        self.num_combinations = len(self.combinations)
        self.log_likelihoods = np.zeros(self.num_combinations)
        self.models = []
        
        for i,pair in enumerate(self.combinations):
            
            # A valid discount factor is defined over (0,1]
            assert (pair[0] <= 1.0 and pair[0] > 0.0)
            assert (pair[1] <= 1.0 and pair[0] > 0.0)
            
            ffbs_model = FFBS(F,G,Y,m0,C0,s0=s0,
                              evo_discount_factor = [pair[0]],
                              obs_discount_factor = pair[1])
            
            ffbs_model.forward_filter()
            self.models.append(ffbs_model)
            self.log_likelihoods[i] = ffbs_model.ll_sum
        
        # We pick the model with the highest likelihood
        best_index = np.argmax(self.log_likelihoods)
        best = self.combinations[best_index]
        self.best_evo   = best[0]
        self.best_obs   = best[1]
        self.best_model = self.models[best_index]

