import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import norm
from scipy.stats import t as student_t

class FFBS_sample(object):
    """An FFBS_sample object is used as a sampling step in a PyMC3 model 
    in order to draw samples of the DLM state vector conditional on 
    emissions/observations and the various DLM structures F,G,W,V."""

    def __init__(self,vars,F,G,r,n,observationsVarName='observations',
                 check_cov_pd = False):
               
        self.F = F
        self.G = G
        self.r = r
        self.n = n
        
        # The attribute 'vars' must be set for PyMC3 to 
        # handle this step object appropriately.
        self.vars = vars
        
        # This is the string name given to the variable representing 
        # the emissions in the DLM. Because we might call this 
        # different things in the larger PyMC3 model, it's handy
        # to be able to adjust this.
        self.observationsVarName = observationsVarName
        self.generates_stats = False
        
        # This flag controls whether we check our backward simulation covariances 
        # for positive semi-definiteness.
        self.check_cov_pd = check_cov_pd
        

    def step(self,estimate):
        
        F = self.F
        G = self.G
         
        # PyMC3 may try to handle the scalar variances as log-transformed variables.
        # Here, we specify that the evolution and observation variances are known, 
        # fixed, and static.
        try:
            W  = np.identity(self.n)*estimate['w']
        except KeyError:
            W  = np.identity(self.n)*np.exp(estimate['w_log__'])
        try: 
            V  = np.identity(self.r)*estimate['v']
        except KeyError:
            V  = np.identity(self.r)*np.exp(estimate['v_log__'])
            
        m0 = estimate['m0']
        
        # This is a hack to easily set the prior covariance.
        C0 = W.copy()
        Y  = estimate[self.observationsVarName]

        self.ffbs = FFBS_known_variance(F,G,W,V,Y,m0,C0,check_cov_pd = self.check_cov_pd)
        self.ffbs.forward_filter()
        self.ffbs.backward_smooth()
        new_state = self.ffbs.backward_sample()
        new_estimate = estimate.copy()
        new_estimate['state'] = new_state

        return new_estimate


class FFBS(object):
    def __init__(self,F,G,V,Y,m0,C0,nancheck = True,check_cov_pd=False,evolution_discount=True,deltas = [0.95],W=None,unknown_obs_var = False,prior_s = 1.0,suppress_warn = False,dynamic_G = False,
                delta_v = 0.98):
        
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
         
        self.check_cov_pd         = check_cov_pd        # Do we want to warn if covariance is not positive definite?
        self.evolution_discount   = evolution_discount  # Discount factor for state evolution variance
        self.unknown_obs_var      = unknown_obs_var     # Is the observational variance observed?
        self.suppress_warn        = suppress_warn       # Warning for G not being singular
        self.dynamic_G            = dynamic_G           # Is G allowed to vary over time?
        self.delta_v              = delta_v             # Discount factor for observational variance
        self.prior_s              = prior_s             # Prior point estimate of observational variance
        
        self.is_filtered          = False
        self.is_backward_smoothed = False
        self.nancheck             = nancheck # Determines whether we check to see if NaNs have appeared
 
        # The dimension of the observed series needs to be determined here.
        # Most of the code will only work for the univariate case; the 
        # multivariate version isn't operational yet.
        try:
            r = Y.shape[1]
        except IndexError:
            r = 1
               
        self.T = Y.shape[0]   # Number of timesteps
        self.r = r            # Observation dimension
        self.n = m0.shape[0]  # State dimension

        # F and G always need to be specified.
        self.F    = F    # Dynamic regression vectors
        self.G    = G    # Evolution matrix
        
        # If the matrix V is not specified (and it usually isn't) the the observational
        # variance is treated as an unknown variable.
        if self.unknown_obs_var: 
            assert V is None
            self.prior_s = prior_s
                  
        else:
            self.V    = V    # Static observation variance with dimension [r,r]
            
        self.Y    = Y    # Observations with dimension [T,r]
        self.m0   = m0   # Prior mean on state vector with dimension [n,1]
        self.C0   = C0   # Prior covariance on state vector with dimensions [n,n]
            
        # We need to make sure that the DLM evolution variance is specified one way or another.
        if evolution_discount:
            
            # For retrospective analysis, G needs to be nonsingular with a 
            # discount approach.
            try:
                np.linalg.cholesky(G) # Fails if G is singular.
            except np.linalg.LinAlgError:
                if not suppress_warn:
                    print 'A discount factor was specified but G is singular. Retrospective analysis will not be reliable.'
        else:
            # Fire of an error if the evolution variance is not specified one way or another.
            if W is None:
                raise ValueError('Neither a discount factor nor evolution variance matrix has been specified.')
            else:
                self.W    = W    # Static state evolution variance matrix
 
        # The discount factors should be passed in as a list of delta values
        # with one delta for each dimension of the state vector.
        # If there is only one delta passed in, then we'll use it as the 
        # global discount factor.
        if len(deltas) == 1:
            self.discount_matrix = np.identity(self.n) *(1.0 /  deltas[0])

        elif len(deltas) == self.n:
            # The diagonal entries of the discount matrix need to be 
            # 1/delta. We just invert a diagonal matrix to get that.
            self.discount_matrix = np.linalg.inv(np.diag(deltas))

        else:
            raise ValueError('Evolution discount factors incorrectly specified.')

            
    def forward_filter(self):

        # These are just for convenience to reduce the number of times that 
        # these static arrays are referenced. The other arrays aren't treated
        # the same because they are frequently manipulated / changed.
        
        F = self.F # Dimensions of [T,n,r]
        Y = self.Y # Dimensions of [T,r]
        T = self.T
        r = self.r
        n = self.n
        
        if self.unknown_obs_var:
            self.gamma_n = np.zeros(T)
            self.s       = np.zeros(T)
            self.s[0]    = self.prior_s
           
        else:
            V = self.V # Dimensions of [r,r]
            
        self.log_likelihood = np.zeros(T) # Per-timestep contribution to LL
        
        self.r = np.zeros(T)       # For unknown obs. variance
        self.e = np.zeros([T,r])   # Forecast error
        self.f = np.zeros([T,r])   # Forecast mean
        self.m = np.zeros([T,n])   # State vector/matrix posterior mean
        self.a = np.zeros([T,n])   # State vector/matrix prior mean
        self.Q = np.zeros([T,r,r]) # Forecast covariance
        self.A = np.zeros([T,n,r]) # Adaptive coefficient vector
        self.R = np.zeros([T,n,n]) # State vector prior variance
        self.C = np.zeros([T,n,n]) # State vector posterior variance
        self.B = np.zeros([T,n,n]) # Retrospective ???

        # Forward filtering
        # For each time step, we ingest a new observation and update our priors
        # to posteriors.
        for t in range(T):
            self.t = t
            
            # If G varies over time, pick out the one we want.
            if self.dynamic_G:
                G = self.G[t]
            else:
                G = self.G # Dimensions of [n,n]

            # If starting out, we use our initial prior mean and prior covariance.
            prior_covariance = self.C0 if t == 0 else self.C[t-1]
            prior_mean       = self.m0 if t == 0 else self.m[t-1]
            self.a[t]   = G.dot(prior_mean)
            
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
            self.f[t]   = F[t].T.dot(self.a[t])
            self.e[t]   = self.Y[t] - self.f[t]
            
            # Next, we calculate the forecast covariance and forecast error.
            # If we don't know the observational variance, then we need
            # to keep track of the prior-to-posterior updating of our distribution
            # over the observational variance.
            if self.unknown_obs_var:
                
                if t == 0:
                    self.Q[t]       = F[t].T.dot(self.R[t]).dot(F[t]) + self.prior_s
                    self.gamma_n[t] = 1.0
                    self.r[t]       = (self.gamma_n[t] + self.e[t]**2 / self.Q[t]) / (self.gamma_n[t] + 1)
                 
                else:
                    self.Q[t]       = F[t].T.dot(self.R[t]).dot(F[t]) + self.s[t-1]
                    self.gamma_n[t] = self.delta_v * self.gamma_n[t-1]+1
                    self.r[t]       = (self.gamma_n[t] + self.e[t]**2 / self.Q[t]) / (self.gamma_n[t] + 1)
                    self.s[t]       = self.r[t] * self.s[t-1]
            
            # In the case where the observational variance is known, the forecast variance
            # is expressed much more succinctly.
            else:
                self.Q[t]   = F[t].T.dot(self.R[t]).dot(F[t]) + V # [r,n] x [n,n] x [n,r]
                  
            # The ratio of R / Q gives us an estimate of the split between
            # prior covariance and forecast covariance.
            if self.unknown_obs_var and t > 0 :
                self.prefactor = self.s[t]/self.s[t-1]
            else:
                self.prefactor = 1.0
            if r == 1:
                self.A[t] = self.R[t].dot(F[t])/np.squeeze(self.Q[t])
                self.C[t] = self.prefactor * (self.R[t] - self.A[t].dot(self.A[t].T)*np.squeeze(self.Q[t]))
            else: # This branch is probably broken as I have not tested any of it for a mv case.
                self.A[t] = self.R[t].dot(F[t]).dot(np.linalg.inv(self.Q[t]))
                self.C[t] = self.prefactor * (self.R[t] - self.A[t].dot(self.Q[t]).dot(self.A[t].T))
            
            # The posterior mean over the state vector is a weighted average 
            # of the prior and the error, weighted by the adaptive coefficient.            
            self.m[t,:]   = self.a[t]+self.A[t].dot(self.e[t])
            
            # The last thing we do in each loop iteration is tabulate the current
            # step's contribution to the overall log-likelihood.
            if self.unknown_obs_var:
                self.log_likelihood[t] = student_t.logpdf(self.e[t], self.gamma_n[t-1], scale=np.sqrt(self.Q[t]))
            else:
                self.log_likelihood[t] = norm.logpdf(self.e[t], scale=np.sqrt(self.Q[t]))

        if self.nancheck:
            try:
                for array in [self.A,self.C,self.Q,self.m]:
                    assert np.any(np.isnan(array)) == False
                          
            except AssertionError:
                print 'NaN values encountered in forward filtering.'
                
        self.is_filtered = True
        self.mae = np.mean(np.abs(self.e))
        self.r2 = r2_score(self.Y,self.f)
        self.ll_sum = self.log_likelihood.sum()
                                                  
        
    def backward_smooth(self):
        """ This method is used to compute retrospective estimates
        of the DLM state vector after we have made a first pass over the
        data with the forward_filtering method.""""
        
        # TODO: add in retrospective analysis for unknown variance
        
        # None of the necessary estimates required for the BS step will be ready
        # if we haven't already applied the forward filtering.
        try:
            assert self.is_filtered
        except SystemError:
            print('The forward filtering process has not been applied yet.')

        # These are the main quantities we care about.
        # The suffix _r indicates that these are from retrospective analyses.
        # s_r and n_r are used only if the observational variance is constant but unknown.   
        self.m_r =  np.zeros([self.T,self.n])         # Retrospective mean over state distribution
        self.C_r =  np.zeros([self.T,self.n,self.n])  # Retrospective posterior covariance over state
        self.s_r =  np.zeros(self.T)                  # Retrospective smoothed estimate of variance
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
                    
                    if self.unknown_obs_var:
                        # Set smoothed estimate of variance to be 
                        # the last forward-filtered estimate of variance
                        self.s_r[t] = self.s[t]
                        self.n_r[t] = self.gamma_n[t]
                    
                else:
                    self.B[t]   = self.C[t].dot(G.T).dot(np.linalg.inv(self.R[t+1]))
                    self.m_r[t] = self.m[t] + self.B[t].dot(self.m_r[t+1] -  self.a[t+1])
                    self.C_r[t] = self.C[t] + self.B[t].dot(self.C_r[t+1] -  self.R[t+1]).dot(self.B[t].T)
                    if self.unknown_obs_var:
                        self.s_r[t] = ((1.0 - self.delta_v) / self.s[t] + self.delta_v / self.s_r[t+1])**-1
                        self.n_r[t] = (1-self.delta_v) * self.gamma_n[t] + self.delta_v * self.gamma_n[t+1]
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
        self.theta = np.zeros([T,self.n,num_samples])
        
        # TODO: optimize code to vectorize the sample drawing
        for i in range(num_samples):
            self.sample_C = np.zeros([T,self.n,self.n])
            self.sample_m = np.zeros([T,self.n])

            # Depending on whether the observational variance is known or 
            # not, we proceed using one of two different recursive sampling algorithms.
            if self.unknown_obs_var:

                # First, we draw a sample variance by making a gamma draw from the 
                # posterior over the precision and inverting it.
                self.sample_precision = np.random.gamma(self.gamma_n[-1]/2.0,scale = 2.0 / (self.gamma_n[-1]*self.s[-1]))
                self.sample_v = 1.0 / sample_precision

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
                    self.sample_C[t]  = self.sample_v * (self.C[t] - self.sample_B.dot(self.sample_C[t] - self.R[t+1]).dot(self.sample_B.T)) / self.s[t]
                    self.sample_m[t]  = self.m[t] + self.sample_B.dot(self.theta[t+1,:,i] - self.a[t+1])
                    self.theta[t,:,i] = np.random.multivariate_normal(self.sample_m[t], self.sample_C[t])

            else:
                # The backward sampling is much simpler if the observational variance is known.
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

        


