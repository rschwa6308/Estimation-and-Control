import numpy as np

from ..filters.filter_base import GaussianBeliefFilter
from ..systems import DifferentiableSystemModel


class ExtendedKalmanFilter(GaussianBeliefFilter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    allowing for potentially non-linear systems, with potentially non-additive zero-mean gaussian noise models.

    System model and measurement model are linearized at the current state estimate, and then linear Kalman estimation is applied.
    """
    # NOTE: this type hint should be the intersection of DifferentiableSystemModel and GaussianSystemModel,
    # but python doesn't currently support intersection types
    def __init__(self, system: DifferentiableSystemModel): 
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
