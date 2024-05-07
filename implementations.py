import numpy as  np
from scipy.spatial.distance import cdist 
from scipy.special import kv, gamma


from sklearn.cluster import KMeans
from auxfunctions import *


matern_model = lambda t, nu: np.vectorize(lambda x: 2**(1-nu) / gamma(nu) * (t*x)**nu * kv(nu, t*x))

def cov_matrix(cov_model, X_1, X_2, rho=1):
    return rho * np.nan_to_num(cov_model(cdist(X_1,X_2)), nan=1)

def k(x, X_1, X_2, cov_1, cov_12, rho_12):
    return np.concatenate([cov_matrix(cov_1, X_1, x), cov_matrix(cov_12, X_2, x, rho_12)])

def K(X_1, X_2, cov_1, cov_2, cov_12, rho_12):
    return np.block([[cov_matrix(cov_1, X_1, X_1), cov_matrix(cov_12, X_1, X_2, rho_12)], 
                     [cov_matrix(cov_12, X_2, X_1, rho_12), cov_matrix(cov_2, X_2, X_2)]])

def K_ij(X_1_i, X_1_j, X_2_i, X_2_j, cov_1, cov_2, cov_12, rho_12):
    return np.block([[cov_matrix(cov_1, X_1_i, X_1_j), cov_matrix(cov_12, X_1_i, X_2_j, rho_12)], 
                     [cov_matrix(cov_12, X_2_i, X_1_j, rho_12), cov_matrix(cov_2, X_2_i, X_2_j)]])



def gen_observations(d, n_1, n_2, sup, nu_1, theta_1, nu_2, theta_2, nu_12, theta_12, rho_12):
    X_1 = np.random.uniform(0, sup, (n_1, d))
    X_2 = np.random.uniform(0, sup, (n_2, d))
    
    sigma = K(X_1, X_2, matern_model(theta_1, nu_1), matern_model(theta_2, nu_2), matern_model(theta_12, nu_12), rho_12)
    Y = np.linalg.cholesky(sigma) @ np.random.normal(0, 1, (n_1+n_2))

    return X_1, X_2, Y[:n_1], Y[n_1:] # (X_1, X_2, Y_1, Y_2)

def gen_A(X, n_clusters):
    KM = KMeans(n_clusters=n_clusters).fit(X)
    A = [[] for _ in range(n_clusters)]
    for i, label in enumerate(KM.labels_):
        A[label].append(i)
    return A


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

def co_kriging(x, X_1, X_2, Y_1, Y_2, cov_1, cov_12, rho_12, sigma):
    'si x es un conjunto de puntos, entrega las predicciones de Y_1'
    c = k(x, X_1, X_2, cov_1, cov_12, rho_12)    
    return c.T @ np.linalg.solve(sigma, np.concatenate([Y_1, Y_2]))

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
        
        self.cov = matern_model(self.theta, self.nu) if cov is None else cov 
        
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
    def __init__(self, X_1, X_2, Y_1, Y_2, theta_1, theta_2, theta_12, nu_1, nu_2, nu_12, rho_12):
        check_consistent_dimension(1, X_1, X_2)
        check_consistent_dimension(0, X_1, Y_1)
        check_consistent_dimension(0, X_2, Y_2)
        self.X_1, self.X_2 = X_1, X_2
        self.Y_1, self.Y_2 = Y_1, Y_2
        self.d = self.X_1.shape[1]
        self.n_1, self.n_2 = self.X_1.shape[0], self.X_2.shape[0]
        
        check_positive_elements(theta_1, theta_2, theta_12, nu_1, nu_2, nu_12)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_12 = theta_12
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu_12 = nu_12
        self.rho_12 = rho_12
        
         
        self.ZZZ = None
        
    def ncok(self, x, A_1, A_2):
        check_consistent_dimension(0, A_1, A_2)
                     
        Y = np.concatenate([self.Y_1, self.Y_2])
        A = [[x,y] for x,y in zip(A_1, A_2)]
        shift_n = self.n_1
        k_x_Xs = k(x, self.X_1, self.X_2, matern_model(self.theta_1, self.nu_1), matern_model(self.theta_12, self.nu_12), self.rho_12)
        
        
        lens_A = [sum(len(subsublist) for subsublist in sublist) for sublist in A]
        cumsum_lens_A = np.cumsum(lens_A)
        
        if self.ZZZ is None:
            self.gen_big_matrix(A, shift_n, lens_A, cumsum_lens_A)
        
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
    
    def gen_big_matrix(self, A, shift_n, lens_A, cumsum_lens_A):
        row, self.ZZZ = 0, np.zeros((sum(lens_A), sum(lens_A)))
        for i in range(len(A)):
            column=cumsum_lens_A[i]
            for j in range(i+1, len(A)):
                self.ZZZ[row:(row+lens_A[i]), column:(column+lens_A[j])] = K_ij(self.X_1[A[i][0]], self.X_1[A[j][0]], 
                                                                                self.X_2[A[i][1]], self.X_2[A[j][1]],
                                                                                matern_model(self.theta_1, self.nu_1),
                                                                                matern_model(self.theta_2, self.nu_2),
                                                                                matern_model(self.theta_12, self.nu_12),
                                                                                self.rho_12)
                column+=lens_A[j]
            row+=lens_A[i]
        self.ZZZ += self.ZZZ.T
        pivot = 0
        for i in range(len(A)):
            self.ZZZ[pivot:(pivot+lens_A[i]), pivot:(pivot+lens_A[i])] = K_ij(self.X_1[A[i][0]], self.X_1[A[i][0]], 
                                                                              self.X_2[A[i][1]], self.X_2[A[i][1]],
                                                                              matern_model(self.theta_1, self.nu_1),
                                                                              matern_model(self.theta_2, self.nu_2),
                                                                              matern_model(self.theta_12, self.nu_12),
                                                                              self.rho_12)
            pivot+=lens_A[i]
    
    def predict(self, X_test, A_1, A_2):
        check_consistent_dimension(0, A_1, A_2)
        return np.array([self.ncok(X_test[[i]], A_1, A_2) for i in range(len(X_test))])
    
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