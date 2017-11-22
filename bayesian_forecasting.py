import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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
    def __init__(self,F,G,V,Y,m0,C0,nancheck = True,check_cov_pd=False,evolution_discount=True,deltas = [0.95],W=None,unknown_obs_var = False,prior_s = 1.0,suppress_warn = False,dynamic_G = False):
        
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
            
        
       
        
        self.check_cov_pd         = check_cov_pd
        self.evolution_discount   = evolution_discount
        self.unknown_obs_var      = unknown_obs_var
        self.suppress_warn        = suppress_warn
        self.dynamic_G            = dynamic_G
        
        self.is_filtered          = False
        self.is_backward_smoothed = False
        self.nancheck             = nancheck # Determines whether we check to see if NaNs have appeared
        
        T = Y.shape[0]
        try:
            r = Y.shape[1]
        except IndexError:
            r = 1
               
        self.T = T            # Number of timesteps
        self.r = r            # Observation dimension
        self.n = m0.shape[0]  # State dimension

        # F and G always need to be specified.
        self.F    = F    # Dynamic regression vectors
        self.G    = G    # Evolution matrix
        
        if self.unknown_obs_var: 
            assert V is None
            self.prior_s = prior_s
                  
        else:
            self.V    = V    # Static observation variance with dimension [r,r]
            
        self.Y    = Y    # Observations with dimension [T,r]
        self.m0   = m0   # Prior mean on state vector with dimension [n,1]
        self.C0   = C0   # Prior covariance on state vector with dimensions [n,n]
            
        # We need to make sure that the DLM evolution variance is specified one way or another.
        if not evolution_discount:
            if W is None:
                raise ValueError('Neither a discount factor nor evolution variance matrix has been specified.')
            else:
                self.W    = W    # Static state evolution variance matrix
                
        else:
            # For retrospective analysis, G needs to be nonsingular with a 
            # discount approach.
            try:
                np.linalg.cholesky(G) # Fails if G is singular.
            except np.linalg.LinAlgError:
                if not suppress_warn:
                    print 'A discount factor was specified but G is singular. Retrospective analysis will not be reliable.'
                
     
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
        
        if not self.unknown_obs_var: 
            V = self.V # Dimensions of [r,r]
        else:
            self.gamma_n = np.zeros(T)
            self.s       = np.zeros(T)
        
        self.e = np.zeros([T,r])   # Forecast error
        self.Q = np.zeros([T,r,r]) # Forecast covariance
        self.f = np.zeros([T,r])   # Forecast mean
        self.m = np.zeros([T,n]) # State vector/matrix posterior mean
        self.a = np.zeros([T,n]) # State vector/matrix prior mean
        self.A = np.zeros([T,n,r]) # Adaptive coefficient vector
        self.R = np.zeros([T,n,n]) # State vector prior variance
        self.C = np.zeros([T,n,n]) # State vector posterior variance
        self.B = np.zeros([T,n,n]) # Retrospective ???
      
        # Forward filtering
        # For each time step, we ingest a new observation and update our priors
        # to posteriors.
        
        # If G varies over time, pick out the one we want.
        for t in range(T):
            self.t = t
            if self.dynamic_G:
                G = self.G[t]
            else:
                G = self.G # Dimensions of [n,n]

            # If starting out, we use our initial prior mean and prior covariance.
            prior_covariance = self.C0 if t == 0 else self.C[t-1]
            prior_mean       = self.m0 if t == 0 else self.m[t-1]
           
            self.a[t]   = G.dot(prior_mean)
            
            if self.evolution_discount:
                self.R[t]   = G.dot(prior_covariance).dot(G.T).dot(self.discount_matrix)
            else:
                self.R[t]   = G.dot(prior_covariance).dot(G.T) + self.W
 
            # The one-step forecast 'f' involves the product of our regression 
            # matrix F and the state vector.
            self.f[t]   = F[t].T.dot(self.a[t])
            self.e[t]   = self.Y[t] - self.f[t]
            

            # Next, we calculate the forecast covariance and forecast error.
            if self.unknown_obs_var:
                if t == 0:
                    self.Q[t]       = F[t].T.dot(self.R[t]).dot(F[t]) + self.prior_s
                    self.gamma_n[t] = 1.0
                    self.s[t]       = self.prior_s 
                    
                else:
                    self.Q[t]       = F[t].T.dot(self.R[t]).dot(F[t]) + self.s[t-1]
                    self.gamma_n[t] = self.gamma_n[t-1]+1
                    self.s[t]       = self.s[t-1] + self.s[t-1] / self.gamma_n[t] * (self.e[t]**2/self.Q[t] - 1)
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
            else:
                self.A[t] = self.R[t].dot(F[t]).dot(np.linalg.inv(self.Q[t]))
                self.C[t] = self.prefactor * (self.R[t] - self.A[t].dot(self.Q[t]).dot(self.A[t].T))
            
            # The posterior mean over the state vector is a weighted average 
            # of the prior and the error, weighted by the adaptive coefficient.            
            self.m[t,:]   = self.a[t]+self.A[t].dot(self.e[t])


        if self.nancheck:
            try:
                for array in [self.A,self.C,self.Q,self.m]:
                    assert np.any(np.isnan(array)) == False
                          
            except AssertionError:
                print 'NaN values encountered in forward filtering.'
                
        self.is_filtered = True
        self.mae = np.mean(np.abs(self.e))
        self.r2 = r2_score(self.Y,self.f)
        
    def backward_smooth(self):
        
        # TODO: add in retrospective analysis for unknown variance
        

        # None of the necessary estimates required for the BS step will be ready
        # if we haven't already applied the forward filtering.
        try:
            assert self.is_filtered
        except SystemError:
            print('The forward filtering process has not been applied yet.')


        # Backward smoothing
        self.a_r =  np.zeros([self.T,self.n])         # Retrospective mean over state distribution
        self.R_r =  np.zeros([self.T,self.n,self.n])  # Retrospective posterior covariance over state
  
  
        for t in range(self.T-1,-1,-1):
                # Unlike in the case of an unknown evolution matrix with discounting,
                # we don't need the inverse of G or G transpose.
                if self.dynamic_G:
                    G       = self.G[t+1]
                                      
                else:
                    G       = self.G 
                    
                if t == ( self.T-1):
                    self.a_r[t] = self.m[t] 
                    self.R_r[t] = self.C[t]
                    
                else:
                    self.B[t]   = self.C[t].dot(G.T).dot(np.linalg.inv(self.R[t+1]))
                    
                    self.a_r[t] = self.m[t] -  self.B[t].dot(self.a[t+1] -  self.a_r[t+1])
                    self.R_r[t] = self.C[t] -  self.B[t].dot(self.R[t+1] -  self.R_r[t+1]).dot(self.B[t].T)
        
        if self.nancheck:
            for array in [self.a_r,self.R_r,self.B]:
                try:
                    assert np.any(np.isnan(array)) == False

                except:
                    print 'NaN values encountered in backward smoothing.'

    def backward_sample(self):
        
        T = self.T
        # Simulated historical value of state
        self.simulated_state = np.zeros([T,self.n])

        # Simulate the end time point from the posterior for T
        # Accessing with the index T-1 gets the last time step
        
        self.simulated_state[T-1] = np.random.multivariate_normal(self.a_r[T-1],self.R_r[T-1])

        # Counts down from the second-to-last until the very first timestep
        for t in range(T-2,-1,-1):
            
            simulation_mean = self.m[t] - self.B[t].dot(self.simulated_state[t+1]-self.a[t+1])
            simulation_cov  = self.C[t] - self.B[t].dot(self.R[t+1]).dot(self.B[t].T)
            
            if self.check_cov_pd:
                try:
                    np.linalg.cholesky(simulation_cov)
                except np.linalg.linalg.LinAlgError:
                    print 'Simulation covariance matrix is not positive definite. Printing simulation covariance,'
                    print simulation_cov
                    print self.B[t]
                    print self.R[t+1]
            self.simulated_state[t] = np.random.multivariate_normal(simulation_mean,simulation_cov)

        return self.simulated_state
    
    
    def pred_vs_obs_plot(self,figsize =(6,6) ):
        """ Wrapper for scatter plot showing the predicted
        versus observed values"""
        
        assert self.is_filtered
        
        low = min([np.min(self.f),np.min(self.Y)]) * 0.9
        high = max([np.max(self.f),np.max(self.Y)])* 1.1
        plt.figure(figsize = figsize )
        plt.scatter(self.Y,self.f,color='k')
        plt.ylabel('Predicted',fontsize = 14)
        plt.xlabel('Observed',fontsize = 14)
        plt.xlim([low,high])
        plt.ylim([low,high])
        plt.plot([low,high],[low,high],linestyle='--',color = 'k')
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

        


