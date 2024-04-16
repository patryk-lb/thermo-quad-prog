#------------------------------------------------------------------------------+
#
#   Simple implementation of Support Vector Machine algorithm
#   2024 (April)
#
#   Inspired by https://domino.ai/blog/fitting-support-vector-machines-quadratic-programming
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import cvxopt

from numpy import linalg

#--- ANCILLARY FUNCTIONS  -----------------------------------------------------+

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))
    
#--- MAIN ---------------------------------------------------------------------+

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
       
    # Train model using dataset (X, y)
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Define Gram matrix using the kernel
        G = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                G[i,j] = self.kernel(X[i], X[j])

        Q = cvxopt.matrix(np.outer(y,y) * G)
        c = cvxopt.matrix(-1 * np.ones(n_samples))
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        # Hard SVM model (positivity constraint)
        if self.C is None:
            B = cvxopt.matrix(np.diag(-1 * np.ones(n_samples)))
            h = cvxopt.matrix(np.zeros(n_samples))
        
        # Soft SVM model  (positivity constraint + nonzero error constraint)
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            B = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem
        solution = cvxopt.solvers.qp(Q, c, B, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        
        # Support vectors have non zero lagrange multipliers
        S = (a > 1e-5).flatten()
        
        # Extract support vectors
        self.a = a[S]
        self.sv = X[S]
        self.sv_y = y[S]

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        # Intercept
        self.b = np.mean(y[S] - np.dot(X[S], (self.w).reshape(-1,1)))
        
    # Predict the class of data points X
    def predict(self, X):
        if self.w is not None:
            projection = np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            
            projection = y_predict + self.b
        
        return np.sign(projection)