import numpy as np

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
    def __init__(self,F,G,V,Y,m0,C0,nancheck = True,check_cov_pd=False,evolution_discount=True,deltas = [0.95],W=None):
        
        # The convention we are using is that F  must be specified as a sequence
        # of [n,r] arrays respectively.
        # The first dimension runs over time and so we must make sure that the
        # arrays fed in have 3 dimensions. Otherwise, we'll run into problems
        # down the road.
        assert len(F.shape) == 3
        
        # We'll also assume that G is fixed over time.
        assert len(G.shape) == 2
        
       
        
        self.check_cov_pd         = check_cov_pd
        self.evolution_discount   = evolution_discount
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

        # All of these need to be defined for the standard DLM (no discounting, known observational variance)
        # to work.
        self.F    = F    # Dynamic regression vectors 
        self.G    = G    # Static evolution matrix
        
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
        G = self.G # Dimensions of [n,n]
        F = self.F # Dimensions of [T,n,r]
        V = self.V # Dimensions of [r,r]
        Y = self.Y # Dimensions of [T,r]
        

        T = self.T
        r = self.r
        n = self.n


        self.e = np.zeros([T,r])   # Forecast error
        self.Q = np.zeros([T,r,r]) # Forecast covariance
        self.f = np.zeros([T,r])   # Forecast mean
        self.m = np.zeros([T,n]) # State vector/matrix posterior mean
        self.a = np.zeros([T,n]) # State vector/matrix prior mean
        self.A = np.zeros([T,n,r]) # Adaptive coefficient vector
        self.R = np.zeros([T,n,n]) # State vector prior variance
        self.C = np.zeros([T,n,n]) # State vector posterior variance
        self.B = np.zeros([T,n,n]) # Retrospective ???
        # Recall that F should have dimensions [T,n,r] so that the product F[t,:,:]'x is an r-vector

        # Forward filtering
        # For each time step, we ingest a new observation and update our priors
        # to posteriors.
        for t in range(T):

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

            # Next, we calculate the forecast covariance and forecast error.
            self.Q[t]   = F[t].T.dot(self.R[t]).dot(F[t]) + V # [r,n] x [n,n] x [n,r]
            self.e[t]   = Y[t]-self.f[t]              
            
            # The ratio of R / Q gives us an estimate of the split between
            # prior covariance and forecast covariance.
            if r == 1:
                self.A[t] = np.squeeze(self.R[t].dot(F[t])/np.squeeze(self.Q[t]))[:,np.newaxis]
                self.C[t] = self.R[t] - self.A[t].dot(self.A[t].T)*np.squeeze(self.Q[t])
            else:
                self.A[t] = self.R[t].dot(F[t]).dot(np.linalg.inv(self.Q[t]))
                self.C[t] = self.R[t] - self.A[t].dot(self.Q[t]).dot(self.A[t].T)
            
            # The posterior mean over the state vector is a weighted average 
            # of the prior and the error, weighted by the adaptive coefficient.            
            self.m[t,:]   = self.a[t]+self.A[t].dot(self.e[t])


        if self.nancheck:
            try:
                assert np.any(np.isnan(self.A)) == False
                assert np.any(np.isnan(self.m)) == False
                assert np.any(np.isnan(self.C)) == False
                assert np.any(np.isnan(self.Q)) == False
            except:
                print 'NaN values encountered in forward filtering.'
        self.is_filtered = True
        
    def backward_smooth(self):
        
        G = self.G

        # None of the necessary estimates required for the BS step will be ready
        # if we haven't already applied the forward filtering.
        try:
            assert self.is_filtered
        except SystemError:
            print('The forward filtering process has not been applied yet.')


        # Backward smoothing
        self.a_r =  np.zeros([self.T,self.n])         # Retrospective mean over state distribution
        self.R_r =  np.zeros([self.T,self.n,self.n])  # Retrospective posterior covariance over state

        # We start out by assuming that the mean/covariance for the final timestep is
        # given by the estimate of mean/covariance we obtained at the end of the 
        # forward pass
        
        # If we use discounting for the state evolution, then we follow 
        # retrospective analysis per the equations in 4.3.6 of Prado & West.
        if self.evolution_discount:
            G_inv = np.linalg.inv(G)
            G_T_inv = np.linalg.inv(G.T)
            
            for t in range( self.T-1,-1,-1):
                self.a_r[t] =  self.m[t] if t==( self.T-1) else  (1.0 - self.discount_matrix).dot(self.m[t]) + self.discount_matrix.dot(G_inv).dot(self.a_r[t+1])
                self.R_r[t] =  self.C[t] if t==( self.T-1) else  (1.0 - self.discount_matrix).dot(self.C[t])+ np.linalg.multi_dot([self.discount_matrix,self.discount_matrix,G_inv,self.R[t+1],G_T_inv])
        
            
        # Default retrospective smoothing for case of known state innovation matrix.
        else:
            for t in range(self.T-1,-1,-1):
                self.B[t]   =  self.C[t].dot(G.T).dot(np.linalg.inv(self.R[self.T-1])) if t == ( self.T-1) else  self.C[t].dot(G.T).dot(np.linalg.inv(self.R[t+1])) 
                self.a_r[t] =  self.m[t]  if t==( self.T-1) else  self.m[t] -  self.B[t].dot(self.a[t+1]-  self.a_r[t+1])
                self.R_r[t] =  self.C[t] if t==( self.T-1) else  self.C[t] -  self.B[t].dot(self.R[t+1] -  self.R_r[t+1]).dot( self.B[t].T)

        if self.nancheck:
            try:
                assert np.any(np.isnan(self.a_r)) == False
                assert np.any(np.isnan(self.R_r)) == False
                assert np.any(np.isnan(self.B))   == False
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


