import numpy as  np
from scipy.spatial.distance import cdist 

from scipy.special import kv, gamma

def rho_matern(d, nu_1, nu_2, nu_12, theta_1, theta_2, theta_12):
    return (theta_12**2/(theta_1*theta_2))**(d/2) * ( gamma(nu_1+d/2)*gamma(nu_2+d/2) / (gamma(nu_1)*gamma(nu_2)) )**(1/2) * gamma(nu_12)/gamma(nu_12+d/2)

def gen_observation_points(d, n, sup):
    return np.random.uniform(0, sup, (n, d))

def generar_grilla(sqrt_n, sup):
    xx = np.linspace(0,sup,sqrt_n)
    X, Y = np.meshgrid(xx,xx)
    return np.column_stack((X.flatten(), Y.flatten())) #Ordenados de izq a der y de abajo hacia arriba

cauchy_model = lambda beta, nu : lambda x : (1 + x**2 / beta)**(-nu) 

cauchy_model_v2 = lambda beta, nu : lambda x : np.float_power((1 + np.power(x, 2) / beta), -nu)

def matern_model(theta, nu):
    if nu == 1/2:
        return lambda x : np.exp(-theta*x) # nu = 1/2
    elif nu == 3/2:
        return lambda x : np.exp(-theta*x)*(1+theta*x) # nu = 3/2
    elif nu == 5/2:
        return lambda x : np.exp(-theta*x)*(1+theta*x+(theta*x)**2/3) # nu = 5/2
    else:
        return np.vectorize(lambda x: 2**(1-nu) / gamma(nu) * (theta*x)**nu * kv(nu, theta*x))

# Covarianzas de Matérn para nu = n + 1/2
matern_model_n_0 = lambda x, theta : np.exp(-theta*x) # nu = 1/2
matern_model_n_1 = lambda x, theta : np.exp(-theta*x)*(1+theta*x) # nu = 3/2
matern_model_n_2 = lambda x, theta : np.exp(-theta*x)*(1+theta*x+(theta*x)**2/3) # nu = 5/2

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