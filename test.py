import unittest
import bayesian_forecasting as bf
from utilities import *
import pandas as pd
import numpy as np
import numpy.testing as npt

class TestCases(unittest.TestCase):

    def test_ar(self):
        """This test case simulates an AR(3) model over 10^5 timesteps with minimal variance
        and attempts to recover the original autoregression coefficients used to generate the data."""

        coefficients = np.asarray([-0.5,0.2,-0.1])
        order = len(coefficients)
        sigma = 0.05
        initial = 0.1
        length = 1000
        y,F = simulate_and_data_matrix_arp(coefficients,sigma,length,initial)
        G = np.identity(order)
        W = np.identity(order) * sigma
        V = 0.1
        Y = y
        m0 = np.ones(order)*0.5
        C0 = np.identity(order) * 0.25
        ffbs = bf.FFBS(F,G,V,Y,m0,C0,W=W,evolution_discount =False)
        ffbs.forward_filter()
        ffbs.backward_smooth()
        sample = ffbs.backward_sample()
        isOkay = True
        for i in range(order):
            upper95 = np.percentile(ffbs.a[:,i],95)
            lower5 = np.percentile(ffbs.a[:,i],5)
            if not (coefficients[i] > lower5) & (coefficients[i] < upper95):
                isOkay = False
        self.assertTrue(isOkay)
         
    
    def test_cyclic(self):
        """ This test case attempts to model a multidecadal temperature time series 
        with a seasonal DLM of order 12 (corresponding to monthly data).
        The mean absolute deviation (i.e. the forecast error in degrees C)
        should be less than 4.0 for the last 10 years of the record."""
        
        sample_data = parseMopex('./sample_data/brookings.csv')
        sample_data = sample_data.dropna(axis = 0)
        monthly = sample_data.groupby(pd.TimeGrouper('m')).mean()
        ld = np.log(monthly['discharge'])
        temp = monthly['max_temp']
        T = len(temp)
        p = 12
        G = permutation_matrix(p)
        W = 5.0
        V = 5.0
        Y = temp
        F = np.zeros([T,p,1])
        F[:,0,:] = 1.0
        m0 = np.ones(p)
        C0 = np.identity(p)
        ffbs = bf.FFBS(F,G,V,Y,m0,C0,evolution_discount = False,W=W)
        ffbs.forward_filter()
        ffbs.backward_smooth()
        self.assertTrue(np.mean(np.abs(ffbs.e[-120::])) < 4.0)
        
        
    def test_cyclic_discount(self):
        """ This test case is identical to 'test_cyclic' save for 
        specification of an innovation discount factor instead of a 
        matrix W.Rhe mean absolute deviation (i.e. the forecast error in degrees C)
        should be less than 4.0 for the last 10 years of the record."""
        
        sample_data = parseMopex('./sample_data/brookings.csv')
        sample_data = sample_data.dropna(axis = 0)
        monthly = sample_data.groupby(pd.TimeGrouper('m')).mean()
        ld = np.log(monthly['discharge'])
        temp = monthly['max_temp']
        T = len(temp)
        p = 12
        G = permutation_matrix(p)
        V = 5.0
        Y = temp
        F = np.zeros([T,p,1])
        F[:,0,:] = 1.0
        m0 = np.ones(p)
        C0 = np.identity(p) * 5
        ffbs = bf.FFBS(F,G,V,Y,m0,C0,evolution_discount = True,deltas=[0.999])
        ffbs.forward_filter()
        ffbs.backward_smooth()
        mae_error = np.mean(np.abs(ffbs.e))
        self.assertTrue(mae_error < 3.0 and mae_error > 2.0)
        

    def test_poly(self):
        """ This test case considers a simple polynomial growth model in which
        the 2nd order coefficient starts at 0.3. The mean absolute error
        of the estimated 2nd state element compared to the true 2nd state
        element should be less than 5.0 and is typically on the order of 1.0."""
        
        n = 2
        r = 1
        T = 1000
        
        static_F = np.asarray([1,0])
        G = np.asarray([[1.0,1.0],[0.0,1.0]])
        W = np.identity(2) 
        v = 1.0
        m0 = np.ones(2)
        m0[1] = 0.3
        
        states,emissions = univariate_dlm_simulation(static_F,G,W,v,m0,n,r,T)
        F = static_F[np.newaxis,:].repeat(T,axis = 0)[:,:,np.newaxis]
        C0 = np.identity(n)
        polynomial_ffbs = bf.FFBS(F,G,v,emissions,m0,C0,evolution_discount = False,W = W)
        polynomial_ffbs.forward_filter()
        polynomial_ffbs.backward_smooth()
        _ = polynomial_ffbs.backward_sample()
        estimated_states = polynomial_ffbs.m
        mae_error = np.mean(np.abs(estimated_states[:,1] - states[:,1]))
        self.assertTrue(mae_error < 5.0)
        
        
if __name__ == '__main__':
    unittest.main()