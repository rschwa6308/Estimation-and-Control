# Estimation-and-Control
Python reference implementations of common state estimation and control algorithms. See `Examples/` for full example usage. 

 - Filters
   - Kalman Filter
   - Extended Kalman Filter
   - Unscented Kalman Filter
 - Controllers
   - coming soon


Example usage:
```python
# Define a linear double-integrator system
from estimation_and_control.systems import LinearSystemModel

dt = 0.1

A = np.array([ [1, dt],
               [0, 1 ] ])

B = np.array([ [0.5 * dt**2],
               [dt         ] ])

C = np.array([ [1, 0] ])

accel_variance = 0.01
R = B @ B.T * accel_variance

measurement_variance = 0.3
Q = np.array([[measurement_variance]])

double_integrator = LinearSystemModel(A, B, C, R, Q)


# Run a Kalman filter
from estimation_and_control.filters.filters import KalmanFilter

car_KF = KalmanFilter(double_integrator)

# to initialize the filter belief...
car_KF.initialize(initial_mean, initial_covariance)

# to filter a timestep...
car_KF.predict_step(x, u)

# to filter a measurement...
car_KF.update_step(z)

# to read off the state estimate...
print(car_KF.mean, car_KF.covariance)
```