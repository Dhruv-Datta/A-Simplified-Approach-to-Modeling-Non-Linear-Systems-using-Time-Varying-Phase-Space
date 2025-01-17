import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import pandas as pd

Cd = 0.5
rho = 0.2
A = 0.9
r = 2
k = 0.5 * Cd * rho * A * r**3
m = 3

Kp = 7
Ki = 0.1
Kd = 8

max_saturation = 15
min_saturation = -15

setpoint = 2

max_rate_for_input_changes = 0.5

time_input = 5
time_inc = 300
input_leftx_coord = -15
input_rightx_coord = 15
input_lefty_coord = -15
input_righty_coord = 15
arrow_density = 30

x0, y0 = 0, 0
initial_condition = [x0, y0]

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(-np.inf, np.inf)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.output_limits = output_limits
        self.last_output = 0
        self.max_rate = max_rate_for_input_changes

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Limit the rate of change
        rate_limited_output = self.last_output + np.clip(output - self.last_output, -self.max_rate, self.max_rate)
        
        self.last_output = np.clip(rate_limited_output, *self.output_limits)
        self.previous_error = error
        return self.last_output


def vector_field(t, z):
    x, y = z
    alpha = pid.compute(setpoint, x, 0.01)
    
    if y >= 0:
        U = y
        V = alpha - (k/(m*r**2))*y**2
    else:
        U = y
        V = alpha + (k/(m*r**2))*y**2
    return [U, V]

# Create the final PID controller with the optimized parameters
pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, output_limits=(min_saturation, max_saturation))

x = np.linspace(input_leftx_coord, input_rightx_coord, arrow_density)
y = np.linspace(input_lefty_coord, input_righty_coord, arrow_density)
X, Y = np.meshgrid(x, y)

time_span = [0, time_input]
t_eval = np.linspace(0, time_input, time_inc)
solution = solve_ivp(vector_field, time_span, initial_condition, t_eval=t_eval)

time = solution.t
path_x, path_y = solution.y

alpha_values = [pid.compute(setpoint, x, time[1] - time[0]) for x in path_x]

data_list = [{'Time': t, 'Alpha': a, 'X': x, 'Y': y} 
             for t, a, x, y in zip(time, alpha_values, path_x, path_y)]

data = pd.DataFrame(data_list)
data.to_csv('vector_field_data_with_pid.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 10))

initial_alpha = pid.compute(setpoint, x0, 0.01)
U_initial = np.where(Y >= 0,
                     Y / np.sqrt(Y**2 + (initial_alpha - (k/(m*r**2))*Y**2)**2),
                     Y / np.sqrt(Y**2 + (initial_alpha + (k/(m*r**2))*Y**2)**2))
V_initial = np.where(Y >= 0,
                     (initial_alpha - (k/(m*r**2))*Y**2) / np.sqrt(Y**2 + (initial_alpha - (k/(m*r**2))*Y**2)**2),
                     (initial_alpha + (k/(m*r**2))*Y**2) / np.sqrt(Y**2 + (initial_alpha + (k/(m*r**2))*Y**2)**2))

q = ax.quiver(X, Y, U_initial, V_initial, color='blue')

particle, = ax.plot([], [], 'ro', label='Particle')
path_line, = ax.plot([], [], 'r-', label='Path')

ax.axhline(0, color='black', linewidth=1.5)
ax.axvline(0, color='black', linewidth=1.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Animated Particle with PID-controlled Vector Field')
ax.set_xlim(input_leftx_coord, input_rightx_coord)
ax.set_ylim(input_lefty_coord, input_righty_coord)
ax.set_aspect('equal')
ax.legend()

def init():
    particle.set_data([], [])
    path_line.set_data([], [])
    return particle, path_line

time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
coord_text = fig.text(0.5, 0.92, '', ha='center', fontsize=12)

def update(frame):
    current_time = time[frame]
    x_current = path_x[frame]
    y_current = path_y[frame]
    
    alpha_current = alpha_values[frame]

    U_dynamic = np.where(Y >= 0,
                         Y / np.sqrt(Y**2 + (alpha_current - (k/(m*r**2))*Y**2)**2),
                         Y / np.sqrt(Y**2 + (alpha_current + (k/(m*r**2))*Y**2)**2))
    V_dynamic = np.where(Y >= 0,
                         (alpha_current - (k/(m*r**2))*Y**2) / np.sqrt(Y**2 + (alpha_current - (k/(m*r**2))*Y**2)**2),
                         (alpha_current + (k/(m*r**2))*Y**2) / np.sqrt(Y**2 + (alpha_current + (k/(m*r**2))*Y**2)**2))

    q.set_UVC(U_dynamic, V_dynamic)

    particle.set_data([x_current], [y_current])
    path_line.set_data(path_x[:frame+1], path_y[:frame+1])
    
    time_text.set_text(f'Time: {current_time:.2f} s | Alpha: {alpha_current:.2f}')
    coord_text.set_text(f'Coordinates: ({x_current:.2f}, {y_current:.2f})')

    return particle, path_line, time_text, coord_text

ani = FuncAnimation(fig, update, frames=len(time), init_func=init, interval=20, repeat=False)

plt.show()
