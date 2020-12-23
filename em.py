"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    D={}
    n,d=X.shape
    k=mixture.mu.shape[0]
    p=mixture.p + 1/10**16
    mu=mixture.mu
    var=mixture.var
    post = np.ones((n, k)) / k
    for i in range(n):
        D[i]=[]
        for j in range(d):
            if X[i][j]!=0:
                D[i].append(j)
    l=0
    for i in range(n):
        temp=np.log(p)+((X[i,D[i]]-mu[:,D[i]])**2).sum(axis=1)/(-2*var)+(len(D[i])/2)*np.log(1/(2*np.pi*var))       
        lp=temp - logsumexp(temp)
        post[i,:]=np.exp(lp)
        l=l+logsumexp(temp)
    return post,l 



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    ar=np.zeros(X.shape)
    n,d=X.shape
    k=post.shape[1]
    p=post.sum(axis=0)
    p=p/n
    mu=mixture.mu
    var=mixture.var
    for i in range(n):
        for j in range(d):
            if X[i][j]!=0:
                ar[i][j]=1
    s=post.T@ar 
    s=s>1
    for j in range(k):
        mu[j,s[j]] = (post[:, j] @ X / (ar.T@post[:,j]))[s[j]]
        sse=post[:,j]@(((X-mu[j]*ar)**2).sum(axis=1))
        var[j]=sse/(ar.sum(axis=1)@post[:,j])
        if var[j]<min_variance:
            var[j]=min_variance
    return GaussianMixture(mu, var, p)  
        


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    pl=None
    l=None
    while (pl==None or l-pl>(1/10**6)*np.abs(l)):
        pl=l
        post,l=estep(X, mixture)
        mixture = mstep(X, post, mixture, 0.25)   
    return mixture,post,l


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post,l=estep(X,mixture)
    m=mixture.mu
    X_pred=X.copy()
    for i in range(post.shape[0]):
        mask= X[i]==0
        X_pred[i,mask]=post[i].T @ m[:,mask]
    return X_pred    
        
        
