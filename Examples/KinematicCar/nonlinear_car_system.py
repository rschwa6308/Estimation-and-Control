import jax.numpy as jnp


################## Kinematic Car: Non-Linear Drag and Turbulence #################
#                                                                                #
#    - state:   [position, velocity]                                             #
#    - control: [acceleration]                                                   #
#    - dynamics: f(x, u) = [                                                     #
#         position + velocity*dt + (1/2)(acceleration + noise_accel)*dt^2,       #
#         velocity + (acceleration - sign(v) * drag)*dt                          #
#     ] where                                                                    #
#         drag = drag_coeff1*velocity + (drag_coeff2 + noise_drag)*velocity^2    #
#    - measurement: h(x) = [position + noise]                                    #
#                                                                                #
##################################################################################

from estimation_and_control.systems import AutoDiffSystemModel


dt = 0.1
drag_coeff1 = 0.0
drag_coeff2 = 0.4

def dynamics_func(x, u, w):
    pos, vel = x.flatten()
    acc = u.flatten()[0]
    noise_acc, noise_drag = w.flatten()
    drag = drag_coeff1*vel + (drag_coeff2 + noise_drag)*vel**2  # <-- non-linear dependence on v
    return jnp.array([
        [pos + vel*dt + 0.5*(acc + noise_acc)*dt**2],
        [vel + (acc - jnp.sign(vel)*drag)*dt]
    ])

def measurement_func(x, v):
    pos, vel = x.flatten()
    noise = v.flatten()[0]
    return jnp.array([
        [pos + noise]
    ])

# dynamics noise covariance
accel_noise_variance = 0.01
drag_noise_variance = 0.001
R = jnp.array([
    [accel_noise_variance, 0],
    [0, drag_noise_variance]
])

# measurement noise covariance
measurement_variance = 0.3
Q = jnp.array([
    [measurement_variance]
])

nonlinear_car_system = AutoDiffSystemModel(2, 1, 1, dynamics_func, measurement_func, R, Q)
nonlinear_car_system.delta_t = dt
