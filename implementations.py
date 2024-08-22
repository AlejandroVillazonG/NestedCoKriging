import numpy as  np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from auxfunctions import *

def cov_matrix(cov_model, X_1, X_2, rho=1, a=0):
    return rho * np.nan_to_num(cov_model(cdist(X_1+a,X_2)), nan=1)

def k(x, X_1, X_2, cov_1, cov_12, rho_12, a=0):
    return np.concatenate([cov_matrix(cov_1, x, X_1).T, cov_matrix(cov_12, x, X_2, rho_12, a).T])

def K(X_1, X_2, cov_1, cov_2, cov_12, rho_12, a=0):
    aux = cov_matrix(cov_12, X_1, X_2, rho_12, a)
    return np.block([[cov_matrix(cov_1, X_1, X_1), aux], 
                     [aux.T, cov_matrix(cov_2, X_2, X_2)]])

def K_ij(X_1_i, X_1_j, X_2_i, X_2_j, cov_1, cov_2, cov_12, rho_12, a=0):
    return np.block([[cov_matrix(cov_1, X_1_i, X_1_j), cov_matrix(cov_12, X_1_i, X_2_j, rho_12, a)], 
                     [cov_matrix(cov_12, X_2_i, X_1_j, rho_12, -a), cov_matrix(cov_2, X_2_i, X_2_j)]])

def gen_observations(X_1, X_2, cov_1, cov_2, cov_12, rho_12):
    n_1 = len(X_1)
    n_2 = len(X_2)
    sigma = K(X_1, X_2, cov_1, cov_2, cov_12, rho_12)
    Y = np.linalg.cholesky(sigma) @ np.random.normal(0, 1, (n_1+n_2))

    return X_1, X_2, Y[:n_1], Y[n_1:] # (X_1, X_2, Y_1, Y_2)

def gen_A(X, n_clusters):
    KM = KMeans(n_clusters=n_clusters).fit(X)
    A = [[] for _ in range(n_clusters)]
    for i, label in enumerate(KM.labels_):
        A[label].append(i)
    return A

def gen_As(X_1, X_2, n_clusters, init="k-means++"):
    KM_1 = KMeans(n_clusters=n_clusters, init=init).fit(X_1)
    KM_2 = KMeans(n_clusters=n_clusters, init=KM_1.cluster_centers_).fit(X_2)

    A_1 = [[] for _ in range(n_clusters)]
    for i, label in enumerate(KM_1.labels_):
        A_1[label].append(i)
        
    A_2 = [[] for _ in range(n_clusters)]
    for i, label in enumerate(KM_2.labels_):
        A_2[label].append(i)
   
    return A_1, A_2

########################### KRIGING ##########################

def kriging(x, X, Y, sigma, cov):
    '''
    x : prediction point
    X : the vector of observation points
    Y : observations
    sigma : covariance matrix of observations
    cov : covariance function cov(h) 
    '''
    return cov_matrix(cov, x, X) @ np.linalg.solve(sigma, Y)

######################### CO-KRIGING #########################

def co_kriging(x, X_1, X_2, Y_1, Y_2, cov_1, cov_12, rho_12, sigma, a=0):
    'si x es un conjunto de puntos, entrega las predicciones de Y_1'
    c = k(x, X_1, X_2, cov_1, cov_12, rho_12, a)    
    return c.T @ np.linalg.solve(sigma, np.concatenate([Y_1, Y_2]))

####################### CO-KRIGING NN ########################

class coKrigingNN:
    def __init__(self, X_1, X_2, Y_1, Y_2, N, cov_family, theta_1, theta_2, theta_12, nu_1, nu_2, nu_12, rho_12, a=0):
        check_consistent_dimension(1, X_1, X_2)
        check_consistent_dimension(0, X_1, Y_1)
        check_consistent_dimension(0, X_2, Y_2)
        # check_consistent_dimension(1, X_1, a)
        self.X_1, self.X_2 = X_1, X_2
        self.Y_1, self.Y_2 = Y_1, Y_2
        self.N = N
        self.cov_family = cov_family
        self.a = a

        check_positive_elements(theta_1, theta_2, theta_12, nu_1, nu_2, nu_12)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_12 = theta_12
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu_12 = nu_12
        self.rho_12 = rho_12
    
    def _predict(self, x):
        indexes_1 = N_nearest_observations_points(self.X_1, x, self.N)
        indexes_2 = N_nearest_observations_points(self.X_2, x, self.N)
        
        sigma = K(self.X_1[indexes_1], self.X_2[indexes_2],
                  self.cov_family(self.theta_1, self.nu_1),
                  self.cov_family(self.theta_2, self.nu_2),
                  self.cov_family(self.theta_12, self.nu_12),
                  self.rho_12, self.a)
        c = k(x, self.X_1[indexes_1], self.X_2[indexes_2],
              self.cov_family(self.theta_1, self.nu_1),
              self.cov_family(self.theta_12, self.nu_12),
              self.rho_12, self.a)    
        return c.T @ np.linalg.solve(sigma, np.concatenate([self.Y_1[indexes_1], self.Y_2[indexes_2]]))
    
    def predict(self, X):
        return np.array([self._predict(X[[i]]) for i in range(len(X))])  

####################### NESTED KRIGING #######################

class NestedKriging:
    '''
    Implements Nested Kriging for predicting outputs based on observations.

    This class provides functionality for predicting outputs using Nested Kriging,
    a method described in the paper "Nested Kriging predictions for datasets with a 
    large number of observations" by Rullière et al. (2018).

    Parameters
    ----------
    X : array-like
        Vector of all observation points.
    Y : array-like
        All observations.
    cov : function
        Isotropic covariance function. C: h -> C(h) 
    theta : float
        Scale parameter for the Matérn covariance model.
    nu : float
        Smoothness parameter for the Matérn covariance model.

    Attributes
    ----------
    X : array-like
        Vector of all observation points.
    Y : array-like
        All observations.
    theta : float or None
        Scale parameter for the Matérn covariance model.
    nu : float or None
        Smoothness parameter for the Matérn covariance model.
    cov : function or None
        Isotropic covariance function. C: h -> C(h) 
    sigma : array-like or None
        Covariance matrix of the observation points.
    ZZZ : array-like or None
        Large matrix used in nested kriging predictions.

    Methods
    -------
    nk(x, A)
        Predicts the output at a given prediction point using nested kriging.
    predict(X_test, A)
        Predicts the output for the given prediction points using nested kriging.
    gen_big_matrix(A, lens_A, cumsum_lens_A)
        Generates a large matrix for nested kriging predictions.
    '''
    def __init__(self, X, Y, cov=None, theta=None, nu=None):
        if len(X) != len(Y):
            raise ValueError(
            "Found input variables with inconsistent numbers of dimensions: %r"
            % [len(X), len(Y)]
            )        
        if theta is not None and nu is not None:
            if theta <= 0 or nu <= 0:
                raise ValueError("Found a non-positive parameter.")
        self.X = X
        self.Y = Y
        self.theta = theta
        self.nu = nu
        
        self.cov = self.cov_family(self.theta, self.nu) if cov is None else cov 
        
        self.sigma = None
        self.ZZZ = None
            
    def nk(self, x, A):
        '''
        Predicts the output at a given prediction point using nested kriging.

        This method implements nested kriging as described in the paper:
        "Nested Kriging predictions for datasets with a large number of observations"
        by Rullière et al. (2018).

        Parameters
        ----------
        x : array-like
            Prediction point.
        A : list of lists
            List of lists containing submodel indexes.

        Returns
        -------
        array
            Predicted output at the given prediction point using nested kriging.
        '''
        
        # if self.sigma is None:
            # self.sigma = cov_matrix(self.cov, self.X, self.X)
        
        k_x_X = cov_matrix(self.cov, self.X, x)    
        lens_A = [len(sublist) for sublist in A]
        cumsum_lens_A = np.cumsum(lens_A)
        
        if self.ZZZ is None:
            self.gen_big_matrix(A, lens_A, cumsum_lens_A)
        
        pivot, C = 0, np.zeros((sum(lens_A), len(A)))
        M = np.zeros((len(A), 1))
        
        for i in range(len(A)):
            C[pivot:(pivot+lens_A[i]), [i]] = np.linalg.solve(self.ZZZ[pivot:(pivot+lens_A[i]), :][:, pivot:(pivot+lens_A[i])], k_x_X[A[i], :])
            M[i] = C[pivot:(pivot+lens_A[i]), [i]].T @ self.Y[A[i]]
            pivot+=lens_A[i]
            
        # cov(M, M) = C.T @ ZZZ @ C
        # cov(M, Y) = diag(C.T @ ZZZ @ C)
        
        K_M = C.T @ self.ZZZ @ C
        return np.diag(K_M) @ np.linalg.solve(K_M, M)
    
    def predict(self, X_test, A):
        '''
        Predicts the output for the given prediction points.

        Parameters
        ----------
        X_test : array-like
            Array containing prediction points.
        A : list of list
            List of lists containing submodel indexes.

        Returns
        -------
        array
            Array containing the predicted outputs for the given prediction points.
        '''
        return np.array([self.nk(X_test[[i]], A) for i in range(len(X_test))])
    
    def gen_big_matrix(self, A, lens_A, cumsum_lens_A):
        '''
        Generates a large matrix for nested kriging predictions.

        This method constructs a large matrix used in nested kriging predictions, 
        composed of blocks of the form cov(X_i, X_j), where X_i represents the points 
        considered in submodel i.

        Parameters
        ----------
        A : list of lists
            List of lists containing submodel indexes.
        lens_A : list
            List containing the lengths of submodel indexes.
        cumsum_lens_A : array-like
            Cumulative sum of lengths of submodel indexes.

        Returns
        -------
        None
        '''
        row, self.ZZZ = 0, np.zeros((sum(lens_A), sum(lens_A)))
        for i in range(len(A)):
            column=cumsum_lens_A[i]
            for j in range(i+1, len(A)):
                self.ZZZ[row:(row+lens_A[i]), column:(column+lens_A[j])] = cov_matrix(self.cov, self.X[A[i]], self.X[A[j]]) # self.sigma[A[i], :][:, A[j]]
                column+=lens_A[j]
            row+=lens_A[i]
        self.ZZZ += self.ZZZ.T
        pivot = 0
        for i in range(len(A)):
            self.ZZZ[pivot:(pivot+lens_A[i]), pivot:(pivot+lens_A[i])] = cov_matrix(self.cov, self.X[A[i]], self.X[A[i]]) # self.sigma[A[i], :][:, A[i]]
            pivot+=lens_A[i]

##################### NESTED CO-KRIGING ######################

class NestedCoKriging:
    '''
    Queremos predecir observaciones de la variable Y_1 utilizando las obs. de ambas.
    '''
    def __init__(self, X_1, X_2, Y_1, Y_2, cov_family, theta_1, theta_2, theta_12, nu_1, nu_2, nu_12, rho_12, a=0):
        check_consistent_dimension(1, X_1, X_2)
        check_consistent_dimension(0, X_1, Y_1)
        check_consistent_dimension(0, X_2, Y_2)
        self.X_1, self.X_2 = X_1, X_2
        self.Y_1, self.Y_2 = Y_1, Y_2
        self.d = self.X_1.shape[1]
        self.n_1, self.n_2 = self.X_1.shape[0], self.X_2.shape[0]
        self.cov_family = cov_family
        self.a = a
        
        check_positive_elements(theta_1, theta_2, theta_12, nu_1, nu_2, nu_12)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_12 = theta_12
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu_12 = nu_12
        self.rho_12 = rho_12
        
         
        self.ZZZ = None
        
    def ncok(self, x, A_1=[], A_2=[], NN=False, n_clusters=None):
        check_consistent_dimension(0, A_1, A_2)

        Y = np.concatenate([self.Y_1, self.Y_2])
        if NN:
            indices_1 = N_nearest_observations_points(self.X_1, x, self.n_1)
            A_1 = indices_1[:(-(self.n_1%n_clusters) if self.n_1%n_clusters!=0 else self.n_1)].reshape((n_clusters, self.n_1//n_clusters)).tolist()
            A_1[-1] += indices_1[-(self.n_1%n_clusters):].tolist() if self.n_1%n_clusters!=0 else []

            indices_2 = N_nearest_observations_points(self.X_2, x, self.n_2)
            A_2 = indices_2[:(-(self.n_2%n_clusters) if self.n_2%n_clusters!=0 else self.n_2)].reshape((n_clusters, self.n_2//n_clusters)).tolist()
            A_2[-1] += indices_2[-(self.n_2%n_clusters):].tolist() if self.n_2%n_clusters!=0 else []

        A = [[x,y] for x,y in zip(A_1, A_2)]
        shift_n = self.n_1
        k_x_Xs = k(x, self.X_1, self.X_2, self.cov_family(self.theta_1, self.nu_1), self.cov_family(self.theta_12, self.nu_12), self.rho_12, self.a)
        
        
        lens_A = [sum(len(subsublist) for subsublist in sublist) for sublist in A]
        cumsum_lens_A = np.cumsum(lens_A)
        
        if self.ZZZ is None or NN:
            self.gen_big_matrix(A, lens_A, cumsum_lens_A)
        
        pivot, C = 0, np.zeros((sum(lens_A), len(A)))
    
        M = np.zeros((len(A), 1))
        
        for i in range(len(A)):
            indexes = A[i][0]+list(shift_n+np.array(A[i][1]))
            
            C[pivot:(pivot+lens_A[i]), [i]] = np.linalg.solve(self.ZZZ[pivot:(pivot+lens_A[i]), pivot:(pivot+lens_A[i])], k_x_Xs[indexes, :])
            
            M[i] = C[pivot:(pivot+lens_A[i]), [i]].T @ Y[indexes]
            
            pivot+=lens_A[i]
            
        # cov(M, M) = C.T @ ZZZ @ C
        # cov(M, Y) = diag(C.T @ ZZZ @ C)
        
        K_M = C.T @ self.ZZZ @ C
        return np.diag(K_M) @ np.linalg.solve(K_M, M)
    
    def gen_big_matrix(self, A, lens_A, cumsum_lens_A):
        row, self.ZZZ = 0, np.zeros((sum(lens_A), sum(lens_A)))
        for i in range(len(A)):
            column=cumsum_lens_A[i]
            for j in range(i+1, len(A)):
                self.ZZZ[row:(row+lens_A[i]), column:(column+lens_A[j])] = K_ij(self.X_1[A[i][0]], self.X_1[A[j][0]], 
                                                                                self.X_2[A[i][1]], self.X_2[A[j][1]],
                                                                                self.cov_family(self.theta_1, self.nu_1),
                                                                                self.cov_family(self.theta_2, self.nu_2),
                                                                                self.cov_family(self.theta_12, self.nu_12),
                                                                                self.rho_12, self.a)
                column+=lens_A[j]
            row+=lens_A[i]
        self.ZZZ += self.ZZZ.T
        pivot = 0
        for i in range(len(A)):
            self.ZZZ[pivot:(pivot+lens_A[i]), pivot:(pivot+lens_A[i])] = K_ij(self.X_1[A[i][0]], self.X_1[A[i][0]], 
                                                                              self.X_2[A[i][1]], self.X_2[A[i][1]],
                                                                              self.cov_family(self.theta_1, self.nu_1),
                                                                              self.cov_family(self.theta_2, self.nu_2),
                                                                              self.cov_family(self.theta_12, self.nu_12),
                                                                              self.rho_12, self.a)
            pivot+=lens_A[i]
    
    def predict(self, X_test, A_1=[], A_2=[], NN=False, n_clusters=None):
        check_consistent_dimension(0, A_1, A_2)
        return np.array([self.ncok(X_test[[i]], A_1, A_2, NN, n_clusters) for i in range(len(X_test))])
    
    def plot_obs(self, figsize=(12, 5)):
        if self.d != 2:
            raise ValueError('La dimensión de los sitios debe ser 2.')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
        ax1.scatter(self.X_1[:, 0], self.X_1[:, 1], c=self.Y_1)
        ax1.set_title(r'$Y_1$')
        scatter2 = ax2.scatter(self.X_2[:, 0], self.X_2[:, 1], c=self.Y_2)
        ax2.set_title(r'$Y_2$')
        fig.tight_layout()
        fig.colorbar(scatter2, ax=[ax1, ax2])