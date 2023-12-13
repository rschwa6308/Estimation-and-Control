import numpy as np
from typing import List, Tuple

from .matrix_square_root import *



class SigmaPointSelector:
    def select_sigma_points(self, mean, cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Select sigma points and weights according to the given distribution. Returns (points, weights_mean, weights_cov)."
        raise NotImplementedError()



class StandardSigmaPointSelector(SigmaPointSelector):
    def __init__(self, alpha=0.1, beta=2.0, kappa=0.0, mat_sqrt_alg=mat_sqrt_cholesky):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.mat_sqrt_alg = mat_sqrt_alg
    
    def select_sigma_points(self, mean, cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = cov.shape[0]    # dimensionality

        # compute covariance sqrt
        cov_sqrt = self.mat_sqrt_alg(cov)

        # compute shell scaling factor
        # lam = self.alpha**2 * (n + self.kappa) - n
        lam = (self.alpha**2 - 1) * n
        shell_scale = np.sqrt(n + lam)

        sigma_points = []
        weights_mean = []
        weights_cov = []

        # include the mean
        sigma_points.append(mean)
        weights_mean.append(lam / (n + lam))
        weights_cov.append(lam / (n + lam) + (1 - self.alpha**2 + self.beta))
        
        # include 2n points in a single shell around the mean
        for i in range(n):
            vec = shell_scale * cov_sqrt[:,i].reshape(-1, 1)
            
            sigma_points.append(mean + vec)
            sigma_points.append(mean - vec)

            weights_mean.append(1 / (2*(n + lam)))
            weights_mean.append(1 / (2*(n + lam)))

            weights_cov.append(1 / (2*(n + lam)))
            weights_cov.append(1 / (2*(n + lam)))

        sigma_points = np.hstack(sigma_points)
        weights_mean = np.array(weights_mean)
        weights_cov = np.array(weights_cov)

        # print("weights_mean:\n", weights_mean)
        # print("weights_cov:\n", weights_cov)

        return (sigma_points, weights_mean, weights_cov)


class MultiShellSigmaPointSelector(SigmaPointSelector):
    def __init__(self, alphas: List[float], beta=2.0, mat_sqrt_alg=mat_sqrt_cholesky):
        self.alphas = alphas
        self.mat_sqrt_alg = mat_sqrt_alg
        self.beta = beta
    
    def select_sigma_points(self, mean, cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = cov.shape[0]    # dimensionality

        # compute covariance sqrt
        cov_sqrt = self.mat_sqrt_alg(cov)

        sigma_points = []
        weights_mean = [0.0]
        weights_cov = [0.0]

        # include the mean
        sigma_points.append(mean)

        for alpha in self.alphas:
            shell_scale = np.sqrt(n) * alpha

            # add weight to center point
            weights_mean[0] += 1 - 1/alpha**2
            weights_cov[0] += 1 - 1/alpha**2 + (1 - alpha**2 + self.beta)

            # include 2n points in a single shell around the mean
            for i in range(n):
                vec = shell_scale * cov_sqrt[:,i].reshape(-1, 1)
                
                sigma_points.append(mean + vec)
                sigma_points.append(mean - vec)

                weights_mean.append(1 / (2*n*alpha**2))
                weights_mean.append(1 / (2*n*alpha**2))

                weights_cov.append(1 / (2*n*alpha**2))
                weights_cov.append(1 / (2*n*alpha**2))

        sigma_points = np.hstack(sigma_points)
        weights_mean = np.array(weights_mean)
        weights_cov = np.array(weights_cov)

        # average across shells
        weights_mean /= len(self.alphas)
        weights_cov /= len(self.alphas)

        # print("weights_mean:\n", weights_mean, np.sum(weights_mean))
        # print("weights_cov:\n", weights_cov, np.sum(weights_cov))

        return (sigma_points, weights_mean, weights_cov)


