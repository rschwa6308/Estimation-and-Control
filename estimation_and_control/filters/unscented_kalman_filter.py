import numpy as np
import scipy

from ..filters.filter_base import GaussianBeliefFilter
from ..systems import GaussianSystemModel
from ..probability.sigma_points import SigmaPointSelector, StandardSigmaPointSelector
from ..probability.transforms import unscented_transform


class UnscentedKalmanFilter(GaussianBeliefFilter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    allowing for potentially non-linear systems, with potentially non-additive zero-mean gaussian noise models.

    System model and measurement model are sampled at so-called 'sigma points' and gaussian distribution is
    fit to their image. Unlike the EKF, the UKF does not require the system to be differentiable.
    """

    def __init__(self, system: GaussianSystemModel, sigma_point_selector: SigmaPointSelector = None):
        super().__init__(system)

        if sigma_point_selector is None:
            sigma_point_selector = StandardSigmaPointSelector()

        self.sigma_point_selector = sigma_point_selector

    def predict_step(self, u):
        # Note: for non-additive noise models, UKF selects sigma points from an "augmented" state + noise space
        augmented_mean = np.concatenate([self.mean, np.zeros((self.system.dynamics_noise_dim, 1))])
        augmented_cov = scipy.linalg.block_diag(self.covariance, self.system.dynamics_noise_cov)

        def augmented_dynamics_model(point):
            x, w = np.split(point, [self.system.state_dim])     # un-augment
            x_prime = self.system.query_dynamics_model(x, u, w)
            return x_prime

        mean_hat, cov_hat, _ = unscented_transform(augmented_dynamics_model, augmented_mean, augmented_cov, self.sigma_point_selector)

        self.mean = mean_hat
        self.covariance = cov_hat

     
    def update_step(self, z):
        # Note: for non-additive noise models, UKF selects sigma points from an "augmented" state + noise space
        augmented_mean = np.concatenate([self.mean, np.zeros((self.system.measurement_noise_dim, 1))])
        augmented_cov = scipy.linalg.block_diag(self.covariance, self.system.measurement_noise_cov)

        def augmented_measurement_model(point):
            x, v = np.split(point, [self.system.state_dim])     # un-augment
            z_hat = self.system.query_measurement_model(x, v)
            return z_hat

        z_hat, cov_zz, cov_xz = unscented_transform(augmented_measurement_model, augmented_mean, augmented_cov, self.sigma_point_selector)

        cov_xz = cov_xz[:self.system.state_dim, :self.system.state_dim]     # chop off augmented noise covariances

        # Kalman update step
        K = cov_xz @ np.linalg.inv(cov_zz)
        self.mean += K @ (z - z_hat)
        self.covariance -= K @ cov_zz @ K.T
