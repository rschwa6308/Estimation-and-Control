from typing import Tuple, Callable
import numpy as np

# import jax.numpy as jnp
import jax



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

    def query_dynamics_model(self, x: np.array, u: np.array, w: np.array = None) -> np.array:
        "Simulate noisy dynamics at state x with control input u. Noise vector w is sampled from self.dynamics_noise_func(), unless a value explicitly provided"
        if w is None:
            w = self.dynamics_noise_func()
        
        return self.dynamics_func(x, u, w)

    def query_measurement_model(self, x: np.array, v: np.array = None) -> np.array:
        "Simulate noisy measurement at state x. Noise vector v is sampled from self.measurement_noise_func(), unless a value is explicitly provided"
        if v is None:
            v = self.measurement_noise_func()
        
        return self.measurement_func(x, v)
    


class GaussianSystemModel(SystemModel):
    "A system model in which both process noise and observation noise are (potentially non-additive) zero-mean gaussians"

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

    def query_dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array, np.array]:
        "Compute jacobian of dynamics model wrt x, wrt u, and wrt w. Return all three in a tuple."

    def query_measurement_jacobian(self, x: np.array) -> Tuple[np.array, np.array]:
        "Compute jacobian of measurement model wrt u and wrt v. Return both in a tuple."


class LinearSystemModel(GaussianSystemModel, DifferentiableSystemModel):
    """
    A time-invariant system model with linear dynamics, linear measurement model, and additive gaussian noises.

    Dynamics: f(x, u) = Ax + Bu + N(0, R)
    Measurement: h(x) = Cx + N(0, Q)

    Note: different texts use different conventions for these parameters
    """

    def __init__(self, A: np.array, B: np.array, C: np.array, R: np.array, Q: np.array):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q

        def dynamics_func(x, u, w):
            return self.A @ x + self.B @ u + w
        
        def measurement_func(x, v):
            return self.C @ x + v

        super().__init__(A.shape[0], B.shape[1], C.shape[0], dynamics_func, measurement_func, R, Q)

    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array]:
        return (self.A, self.B, self.R)
    
    def measurement_jacobian(self, x: np.array) -> np.array:
        return (self.C, self.Q)


class AutoDiffSystemModel(GaussianSystemModel, DifferentiableSystemModel):
    """
    A time-invariant system model with arbitrary dynamics, arbitrary measurement model, and potentially
    non-additive gaussian noises (R, Q). The jacobians are computed at runtime via automatic differentiation (using JAX). 

    The constructor accepts functions for the underlying dynamics and measurement models. Note that (for the sake of consistency) these functions must operate on column vectors:
     - `dynamics_func(x: (1, state_dim), u: (1, control_dim), w: (1, dynamics_noise_dim)) -> (1, state_dim)`
     - `measurement_func(x: (1, state_dim), v: (1, measurement_noise_dim)) -> (1, measurement_dim)`
    
    Moreover, these functions must operate on `jax.numpy` arrays and must use the corresponding methods.
    """

    def __init__(self, state_dim, control_dim, measurement_dim, dynamics_func, measurement_func, dynamics_noise_cov, measurement_noise_cov):
        super().__init__(state_dim, control_dim, measurement_dim, dynamics_func, measurement_func, dynamics_noise_cov, measurement_noise_cov)

        # precompile jacobian evaluation functions
        self.dynamics_func_dx = jax.jit(jax.jacfwd(dynamics_func, argnums=0))       # wrt first arg (x)
        self.dynamics_func_du = jax.jit(jax.jacfwd(dynamics_func, argnums=1))       # wrt second arg (u)
        self.dynamics_func_dw = jax.jit(jax.jacfwd(dynamics_func, argnums=2))       # wrt second arg (w)

        self.measurement_func_dx = jax.jit(jax.jacfwd(measurement_func, argnums=0))     # wrt first arg (x)
        self.measurement_func_dv = jax.jit(jax.jacfwd(measurement_func, argnums=1))     # wrt second arg (v)


    def query_dynamics_jacobian(self, x: np.array, u: np.array, w: np.array) -> Tuple[np.array, np.array]:
        # evaluate jacobians (JAX requires explicit floats)
        F_x = self.dynamics_func_dx(x.astype(float), u.astype(float), w.astype(float))
        F_u = self.dynamics_func_du(x.astype(float), u.astype(float), w.astype(float))
        F_w = self.dynamics_func_dw(x.astype(float), u.astype(float), w.astype(float))

        # remove extra column-vector dimensions
        F_x = F_x.reshape(self.state_dim, self.state_dim)
        F_u = F_u.reshape(self.state_dim, self.control_dim)
        F_w = F_w.reshape(self.state_dim, self.dynamics_noise_dim)

        return (F_x, F_u, F_w)

    def query_measurement_jacobian(self, x: np.array, v: np.array) -> np.array:
        # evaluate jacobian (JAX requires explicit floats)
        H_x = self.measurement_func_dx(x.astype(float), v.astype(float))
        H_v = self.measurement_func_dv(x.astype(float), v.astype(float))

        # remove extra column-vector dimensions
        H_x = H_x.reshape((self.measurement_dim, self.state_dim))
        H_v = H_v.reshape((self.measurement_dim, self.measurement_noise_dim))

        return (H_x, H_v)
