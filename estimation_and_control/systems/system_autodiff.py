import numpy as np
import jax

from estimation_and_control.systems import DifferentiableSystemModel, GaussianSystemModel


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


    def query_dynamics_jacobian(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # evaluate jacobians (JAX requires explicit floats)
        F_x = self.dynamics_func_dx(x.astype(float), u.astype(float), w.astype(float))
        F_u = self.dynamics_func_du(x.astype(float), u.astype(float), w.astype(float))
        F_w = self.dynamics_func_dw(x.astype(float), u.astype(float), w.astype(float))

        # remove extra column-vector dimensions
        F_x = F_x.reshape(self.state_dim, self.state_dim)
        F_u = F_u.reshape(self.state_dim, self.control_dim)
        F_w = F_w.reshape(self.state_dim, self.dynamics_noise_dim)

        return (F_x, F_u, F_w)

    def query_measurement_jacobian(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # evaluate jacobian (JAX requires explicit floats)
        H_x = self.measurement_func_dx(x.astype(float), v.astype(float))
        H_v = self.measurement_func_dv(x.astype(float), v.astype(float))

        # remove extra column-vector dimensions
        H_x = H_x.reshape((self.measurement_dim, self.state_dim))
        H_v = H_v.reshape((self.measurement_dim, self.measurement_noise_dim))

        return (H_x, H_v)
