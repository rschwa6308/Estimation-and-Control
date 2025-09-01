import numpy as np


###################### Kinematic Car: Linear System #####################
#                                                                       #
#    - state:   [position, velocity]                                    #
#    - control: [acceleration]                                          #
#    - dynamics: f(x, u) = [                                            #
#         position + velocity*dt + (1/2)(acceleration + noise)*dt^2,    #
#         velocity + acceleration*dt                                    #
#     ]                                                                 #
#    - measurement: h(x) = [position + noise]                           #
#                                                                       #
#########################################################################


from estimation_and_control.systems import LinearSystemModel


dt = 0.1

A = np.array([ [1, dt],
               [0, 1 ] ])

B = np.array([ [0.5 * dt**2],
               [dt         ] ])

C = np.array([[1, 0]])

accel_variance = 0.01           # process noise
R = B @ B.T * accel_variance

measurement_variance = 0.3      # measurement noise
Q = np.array([[measurement_variance]])

linear_car_system = LinearSystemModel(A, B, C, R, Q)
linear_car_system.delta_t = dt
