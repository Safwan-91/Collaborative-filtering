"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n,d=X.shape
    k=mixture.mu.shape[0]
    p=mixture.p
    mu=mixture.mu
    var=mixture.var
    post=np.zeros([n,k])
    l=0
    for i in range(n):
        temp=p*np.exp(((X[i]-mu)**2).sum(axis=1)/-(2*var))/(2*np.pi*var)**(d/2)
        post[i]=temp/sum(temp)
        l=l+sum(post[i,:]*np.log(temp/post[i,:]))
    return post,l    

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d=X.shape
    k=post.shape[1]
    p=post.sum(axis=0)
    p=p/n
    mu=np.zeros([k,d])
    var=np.zeros(k)
    cost=0
    for j in range(k):
        mu[j,:] = post[:, j] @ X / sum(post[:,j])
        sse=post[:,j]@(((X-mu[j])**2).sum(axis=1))
        var[j]=sse/(d*sum(post[:,j]))
        cost+=sse
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
    while (pl==None or l-pl>(1/10**-6)*np.abs(l)):
        pl=l
        post,l=estep(X, mixture)
        mixture = mstep(X, post)   
    return mixture,post,l
    




