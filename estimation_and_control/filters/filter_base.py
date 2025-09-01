import numpy as np

from ..systems import SystemModel, LinearSystemModel
from ..probability.distributions import ProbabilityDistribution, GaussianDistribution


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
