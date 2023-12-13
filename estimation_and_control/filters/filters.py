import numpy as np
import scipy.linalg

from estimation_and_control.systems import GaussianSystemModel, SystemModel, LinearSystemModel, DifferentiableSystemModel
from estimation_and_control.probability.distributions import ProbabilityDistribution, GaussianDistribution
from estimation_and_control.probability.sigma_points import SigmaPointSelector, StandardSigmaPointSelector
from estimation_and_control.probability.transforms import unscented_transform


class Filter:
    """
    High level representation of the Baye's Filter
    """

    def __init__(self, system: SystemModel):
        self.system = system
        self.belief = ProbabilityDistribution(system.state_dim)

    def predict_step(self, u):
        pass

    def update_step(self, z):
        pass


class GaussianBeliefFilter(Filter):
    """
    A class of filters in which the belief is parametrized as a multivariate gaussian distribution.
    For convenience, `self.belief.mean` and `self.belief.covariance` are forwarded to `self.mean` and `self.covariance`.
    """

    def __init__(self, system: LinearSystemModel):
        super().__init__(system)
        self.belief = GaussianDistribution(
            np.zeros(self.system.state_dim),
            np.eye(self.system.state_dim)
        )
    
    def initialize(self, mean, covariance):
        self.belief.mean = mean
        self.belief.covariance = covariance
    
    @property
    def mean(self):
        return self.belief.mean
    
    @property
    def covariance(self):
        return self.belief.covariance

    @mean.setter
    def mean(self, value):
        self.belief.mean = value
    
    @covariance.setter
    def covariance(self, value):
        self.belief.covariance = value


class KalmanFilter(GaussianBeliefFilter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    restricted to purely linear system models with additive zero-mean gaussian noise models.
    """
    def __init__(self, system: LinearSystemModel):
        super().__init__(system)

    def predict_step(self, u):
        """
        Kalman predict step:
         - µ = Aµ + Bu
         - Σ = AΣA^T + R
        """
        self.mean = self.system.A @ self.mean + self.system.B @ u
        self.covariance = self.system.A @ self.covariance @ self.system.A.T + self.system.R

    def update_step(self, z):
        """
        Kalman update step:
         - µ = µ + K(z - Cµ)
         - Σ = Σ - KCΣ

        where
         - K = ΣC^T(CΣC^T + Q)^-1
        is the so-called "Kalman gain"
        """
        K = self.covariance @ self.system.C.T @ np.linalg.inv(self.system.C @ self.covariance @ self.system.C.T + self.system.Q)

        self.mean += K @ (z - self.system.C @ self.mean)
        self.covariance -= K @ self.system.C @ self.covariance


class ExtendedKalmanFilter(GaussianBeliefFilter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    allowing for potentially non-linear systems, with potentially non-additive zero-mean gaussian noise models.

    System model is linearized at the current state estimate, and then linear Kalman estimation is applied.
    """
    def __init__(self, system: DifferentiableSystemModel):  # NOTE: this type hint should be the intersection of DifferentiableSystemModel and GaussianSystemModel, but python doesn't currently support intersection types
        super().__init__(system)
    
    def predict_step(self, u):
        """
        First, linearize the dynamics model at the current state estimate to obtain A, B, and L
        (respectively F_x, F_u, and F_w).
        Then, perform the Kalman predict step:
         - µ = f(µ, u)
         - Σ = AΣA^T + LRL^T
        """
        # linearize dynamics model
        w_mean = np.zeros((self.system.dynamics_noise_dim, 1))
        A, B, L = self.system.query_dynamics_jacobian(self.mean, u, w_mean)

        # Kalman predict step
        self.mean = self.system.query_dynamics_model(self.mean, u)
        self.covariance = A @ self.covariance @ A.T + L @ self.system.dynamics_noise_cov @ L.T
 

    def update_step(self, z):
        """
        First, linearize the measurement model at the current state estimate to obtain C and M
        (respectively H_x and H_v).
        Then, perform the Kalman update step:
         - µ = µ + K(z - h(µ))
         - Σ = Σ - KCΣ

        where
         - K = ΣC^T(CΣC^T + MQM^T)^-1
        is the so-called "Kalman gain"
        """
        # linearize measurement model
        v_mean = np.zeros((self.system.measurement_noise_dim, 1))

        C, M = self.system.query_measurement_jacobian(self.mean, v_mean)

        # compute Kalman gain
        K = self.covariance @ C.T @ np.linalg.inv(C @ self.covariance @ C.T + M @ self.system.measurement_noise_cov @ M.T)
        
        self.mean += K @ (z - self.system.query_measurement_model(self.mean))
 
        self.covariance -= K @ C @ self.covariance

    

class UnscentedKalmanFilter(GaussianBeliefFilter):
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
