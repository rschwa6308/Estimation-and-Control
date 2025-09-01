import numpy as np

from ..filters.filter_base import GaussianBeliefFilter
from ..systems import LinearSystemModel


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
