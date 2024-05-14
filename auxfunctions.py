import numpy as  np
from scipy.spatial.distance import cdist 

def gen_observation_points(d, n, sup):
    return np.random.uniform(0, sup, (n, d))

def generar_grilla(sqrt_n, sup):
    xx = np.linspace(0,sup,sqrt_n)
    X, Y = np.meshgrid(xx,xx)
    return np.column_stack((X.flatten(), Y.flatten())) #Ordenados de izq a der y de abajo hacia arriba

def N_nearest_observations_points(X, x, N):
    """
    Finds the N nearest points to a given point in a set of points.

    Parameters:
    -----------
    X : numpy array of shape (n, d)
        Set of points, where n is the number of points
        and d is the dimension of the points.

    x : numpy array of shape (1, d)
        Point for which the nearest points in X are to be found.

    N : int
        Number of nearest points to be found.

    Returns:
    --------
    numpy array of shape (N,)
        Contains the indexes of N nearest points to x in X.
    """
    return cdist(x, X).flatten().argsort()[:N]

def check_consistent_dimension(d, *arrays):
    """Check whether all arrays have consistent dimensions along a specified axis.

    This function verifies if all arrays have the same shape or length along a specific dimension.

    Parameters
    ----------
    d : int
        The dimension along which consistency will be checked.
        For example, use d=0 to ensure the same number of observations,
        and d=1 to ensure the same length of observations.
    *arrays : list or tuple of input objects
        Objects that will be checked for consistent dimensions.

    Raises
    ------
    ValueError
        If input variables have inconsistent dimensions along the specified axis.
    """

    dimensions = [(X.shape[d] if d>0 else len(X)) for X in arrays if X is not None]
    if len(np.unique(dimensions)) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of dimensions: %r"
            % [int(l) for l in dimensions]
        )
        
        
def check_positive_elements(*args):
    """Check whether all elements in the provided arguments are greater than 0.

    Parameters
    ----------
    *args : int or float
        Values to be checked for positivity.

    Raises
    ------
    ValueError
        If any provided value is not greater than 0.
    """
    for arg in args:
        if arg <= 0:
            raise ValueError("Found a non-positive parameter.")