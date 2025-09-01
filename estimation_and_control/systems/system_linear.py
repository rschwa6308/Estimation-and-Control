import numpy as np

from estimation_and_control.systems import DifferentiableSystemModel, GaussianSystemModel


class LinearSystemModel(GaussianSystemModel, DifferentiableSystemModel):
    """
    A time-invariant system model with linear dynamics, linear measurement model, and additive gaussian noises.

     - Dynamics: `f(x, u) = Ax + Bu + N(0, R)`
     - Measurement: `h(x) = Cx + N(0, Q)`

    Note: different texts use different conventions for these parameters
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, R: np.ndarray, Q: np.ndarray):
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

    def dynamics_jacobian(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.A, self.B, self.R)
    
    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        return (self.C, self.Q)
