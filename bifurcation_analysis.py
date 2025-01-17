import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from itertools import product
from tqdm import tqdm
import csv
from joblib import Parallel, delayed

Cd = 0.5
rho = 0.2
A = 0.9
r = 2
k = 0.5 * Cd * rho * A * r**3
m = 3

total_time = 200
time_inc = 1000

full_max_rate = 0.5

total_plots = 15

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(-np.inf, np.inf), max_rate=full_max_rate):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.output_limits = output_limits
        self.last_output = 0
        self.max_rate = max_rate

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        rate_limited_output = self.last_output + np.clip(output - self.last_output, -self.max_rate, self.max_rate)
        
        self.last_output = np.clip(rate_limited_output, *self.output_limits)
        self.previous_error = error
        return self.last_output

def system_dynamics(t, state, pid, psi_ref):
    psi, psi_dot = state
    dt = 0.01
    alpha = pid.compute(psi_ref, psi, dt)
    
    if psi_dot >= 0:
        force = alpha - (k / (m * r**2)) * psi_dot**2
    else:
        force = alpha + (k / (m * r**2)) * psi_dot**2
    
    dpsi_dt = psi_dot
    dpsi_dot_dt = force
    return [dpsi_dt, dpsi_dot_dt]

def simulate_system(Kp, Ki, Kd, psi_ref, t_span, y0):
    pid = PIDController(Kp, Ki, Kd, output_limits=(-15, 15))
    t_eval = np.linspace(t_span[0], t_span[1], time_inc)
    sol = solve_ivp(system_dynamics, t_span, y0, args=(pid, psi_ref), t_eval=t_eval, method='RK45')
    return sol.t, sol.y

def check_stability(t, y, psi_ref, epsilon=1e-2, error_threshold=1e-2):
    psi_n = y[0]
    error = psi_ref - psi_n
    tail_fraction = int(0.05 * len(error))
    error_tail = error[-tail_fraction:]
    is_converged = np.all(np.abs(error_tail - error_tail.mean()) < epsilon)
    is_within_threshold = np.all(np.abs(error_tail) < error_threshold)
    return is_converged and is_within_threshold

def save_results_to_csv(results, filename="bifurcation_results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Kp", "Ki", "Kd", "Stable"])
        writer.writerows(results)

def simulate_and_check(Kp, Ki, Kd, psi_ref, t_span, y0):
    t, y = simulate_system(Kp, Ki, Kd, psi_ref, t_span, y0)
    is_stable = check_stability(t, y, psi_ref)
    return (Kp, Ki, Kd, is_stable)

def multidimensional_bifurcation(Kp_range, Ki_range, Kd_range, psi_ref, t_span, y0):
    parameter_space = list(product(Kp_range, Ki_range, Kd_range))
    
    print("Starting multidimensional bifurcation analysis:")
    
    results = Parallel(n_jobs=-1)(
        delayed(simulate_and_check)(Kp, Ki, Kd, psi_ref, t_span, y0) for Kp, Ki, Kd in tqdm(parameter_space, desc="Simulating")
    )
    
    stable_results = [result for result in results if result[3]]
    unstable_results = [result for result in results if not result[3]]
    return stable_results, unstable_results

def plot_trajectories(Kp, Ki, Kd, psi_ref, t_span, y0):
    t, y = simulate_system(Kp, Ki, Kd, psi_ref, t_span, y0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, y[0], label="ψ (angle)")
    ax.plot(t, y[1], label="ψ_dot (angular velocity)")
    ax.set_title(f"Trajectories (Kp={Kp}, Ki={Ki}, Kd={Kd})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State Variables")
    ax.legend()
    plt.show()

def plot_stability_3d(results):
    stable_points = [(Kp, Ki, Kd) for Kp, Ki, Kd, stable in results if stable]
    unstable_points = [(Kp, Ki, Kd) for Kp, Ki, Kd, stable in results if not stable]
    
    stable_points = np.array(stable_points)
    unstable_points = np.array(unstable_points)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(stable_points) > 0:
        ax.scatter(stable_points[:, 0], stable_points[:, 1], stable_points[:, 2], color='green', label='Stable')
    if len(unstable_points) > 0:
        ax.scatter(unstable_points[:, 0], unstable_points[:, 1], unstable_points[:, 2], color='red', label='Unstable')
    
    ax.set_xlabel('$K_p$')
    ax.set_ylabel('$K_i$')
    ax.set_zlabel('$K_d$')
    ax.set_title('Stability in $(K_p, K_i, K_d)$ Space')
    ax.legend()
    plt.show()

def plot_multiple_trajectories(results, psi_ref, t_span, y0, title, max_plots=total_plots):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (Kp, Ki, Kd, _) in enumerate(results[:max_plots]):
        t, y = simulate_system(Kp, Ki, Kd, psi_ref, t_span, y0)
        ax.plot(t, y[0], label=f"Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
    
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ψ (angle)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=psi_ref, color='r', linestyle='--', label='Reference')
    plt.tight_layout()
    plt.show()

def plot_topographical_stability(results):
    # Extract values
    Kp_values = np.array([row[0] for row in results])
    Ki_values = np.array([row[1] for row in results])
    Kd_values = np.array([row[2] for row in results])
    stability_statuses = np.array([1 if len(row) == 4 and row[3] else 0 for row in results])
    
    # Create a regular grid to interpolate the data
    xi = np.linspace(min(Kp_values), max(Kp_values), 100)
    yi = np.linspace(min(Kd_values), max(Kd_values), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate stability values
    from scipy.interpolate import griddata
    zi = griddata((Kp_values, Kd_values), stability_statuses, (xi, yi), method='cubic', fill_value=0)
    
    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    zi = gaussian_filter(zi, sigma=1)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with smooth edges
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis',
                          linewidth=0,
                          antialiased=True,
                          alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel(r'$K_p$')
    ax.set_ylabel(r'$K_d$')
    ax.set_zlabel('Stability')
    ax.set_title('Topographical Stability Visualization')
    
    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    cbar.set_label('Stability (1=Stable, 0=Unstable)')
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.show()

if __name__ == "__main__":
    Kp_range = np.linspace(0.5, 10, 15)
    Ki_range = np.linspace(0.01, 0.1, 4)
    Kd_range = np.linspace(0.5, 10, 15)
    
    psi_ref = 2.0
    t_span = (0, total_time)
    y0 = [0.0, 0.0]
    
    stable_results, unstable_results = multidimensional_bifurcation(Kp_range, Ki_range, Kd_range, psi_ref, t_span, y0)
    
    save_results_to_csv(stable_results + unstable_results)
    
    plot_stability_3d(stable_results + unstable_results)

    plot_topographical_stability(stable_results + unstable_results)
    
    plot_multiple_trajectories(stable_results, psi_ref, t_span, y0, "Stable Systems (10 examples)")
    
    plot_multiple_trajectories(unstable_results, psi_ref, t_span, y0, "Unstable Systems (10 examples)")
