import numpy as np
from robo.models.base_model import BaseModel


class RFE(BaseModel):
    
    def __init__(self, l, sigma_w, *args, **kwargs):
        """
        Random Fourier Expansions
        
        Parameters
        ----------
        l = lambda: regularization parameter
        sigma_w = standard deviation of the Gaussian distribution
        """
        self.l =  l
        self.sigma_w = sigma_w
        
    
    def train(self, X, Y,**kwargs):
        """
        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        """
        self.X = X
        self.Y = Y
        N,D = X.shape

        b =  []
        for i in xrange(0,D):
            bk = np.random.uniform(0,2*np.pi)
            b.append(bk)
        self.b = np.array(b)
        # omega is drawn randomly from a Gaussian distribution
        self.w = np.random.normal(loc=0.0,scale=self.sigma_w, size=(D,D))
        # getting the weights c
        A = np.cos((self.w.dot(X.T)).T+self.b)
        self.cn = (1/((1/N)*(A.T.dot(A)) + np.multiply(self.l,np.eye(D,D)))).dot((1/N)*(A.T.dot(Y)))
        g = (A.dot(self.cn))/np.sqrt(N)
        
        

    def predict(self, X_test, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the specified test point.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,1)
            predictive mean
        np.array(N,1)
            predictive variance
        """
        N,D = X_test.shape
        var = np.zeros(X_test.shape)

        A_test = np.cos((self.w.dot(X_test.T)).T+self.b)
        g = (A_test.dot(self.cn))/np.sqrt(N)
        return g, var
        