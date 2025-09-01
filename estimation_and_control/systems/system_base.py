from typing import Tuple, Callable
import numpy as np

class SystemModel:
    """
    A high-level class representing a dynamic system, including a dynamics model, measurement model, and corresponding noise models.

    Constructor requires the dimensions of the following spaces:
     - `state_dim`: dimension of the state space (x)
     - `control_dim`: dimension of the control input space (u)
     - `measurement_dim`: dimension of the measurement space (z)
     - `dynamics_noise_dim`: dimension of the dynamics noise vector space (w)
     - `measurement_noise_dim`: dimension of the measurement noise vector space (v)
    
    in addition to four callable functions:
     - `dynamics_func`: (x, u, w) -> x
     - `measurement_func`: (x, v) -> z
     - `dynamics_noise_func`: () -> w
     - `measurement_noise_func`: () -> v

    Subclasses include:
     - `LinearSystemModel`
     - `DifferentiableSystemModel`
    """
    def __init__(self,
        state_dim, control_dim, measurement_dim,
        dynamics_noise_dim, measurement_noise_dim,
        dynamics_func, measurement_func,
        dynamics_noise_func, measurement_noise_func,
        delta_t=None
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim

        self.dynamics_noise_dim = dynamics_noise_dim
        self.measurement_noise_dim = measurement_noise_dim

        self.dynamics_func = dynamics_func
        self.measurement_func = measurement_func

        self.dynamics_noise_func = dynamics_noise_func
        self.measurement_noise_func = measurement_noise_func

        self.delta_t = delta_t

    def query_dynamics_model(self, x: np.ndarray, u: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        "Simulate noisy dynamics at state x with control input u. Noise vector w is sampled from self.dynamics_noise_func(), unless a value explicitly provided"
        if w is None:
            w = self.dynamics_noise_func()
        
        return self.dynamics_func(x, u, w)

    def query_measurement_model(self, x: np.ndarray, v: np.ndarray = None) -> np.ndarray:
        "Simulate noisy measurement at state x. Noise vector v is sampled from self.measurement_noise_func(), unless a value is explicitly provided"
        if v is None:
            v = self.measurement_noise_func()
        
        return self.measurement_func(x, v)


class GaussianSystemModel(SystemModel):
    "A system model in which both process noise (Q) and measurement noise (R) are (potentially non-additive) zero-mean gaussians"

    def __init__(self, state_dim, control_dim, measurement_dim, dynamics_func, measurement_func, dynamics_noise_cov, measurement_noise_cov):
        self.dynamics_noise_cov = dynamics_noise_cov
        self.measurement_noise_cov = measurement_noise_cov

        def dynamics_noise_func():
            w = np.random.multivariate_normal(np.zeros(self.dynamics_noise_dim), dynamics_noise_cov)
            return w.reshape((-1, 1))
        
        def measurement_noise_func():
            v = np.random.multivariate_normal(np.zeros(self.measurement_noise_dim), measurement_noise_cov)
            return v.reshape((-1, 1))

        super().__init__(
            state_dim, control_dim, measurement_dim,
            dynamics_noise_cov.shape[0], measurement_noise_cov.shape[0],
            dynamics_func, measurement_func,
            dynamics_noise_func, measurement_noise_func
        )


class DifferentiableSystemModel(SystemModel):
    """
    An abstract class representing a system model with differentiable dynamics and measurement models.
    Such a system is amenable to the EKF.

    Subclasses include:
     - `LinearSystemModel`: special case where jacobians are determined directly from system parametrization 
     - `AutoDiffSystemModel`: jacobians computed through automatic differentiation
     - `SymbDiffSystemModel`: jacobians provided explicitly be user
    """

    def query_dynamics_jacobian(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Compute jacobian of dynamics model wrt x, wrt u, and wrt w. Return all three in a tuple."

    def query_measurement_jacobian(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        "Compute jacobian of measurement model wrt u and wrt v. Return both in a tuple."
