import jax
from jax import random
import jax.numpy as jnp
from jax.experimental.ode import odeint
import control
import matplotlib.pyplot as plt
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
import os
import time

## Simulation 
# System parameters
m1 = 2  # Cart mass [kg]
m2 = 1  # Pendulum mass [kg]
L = 1  # Pendulum length [m]
g = 10 # Gravity [m/s^2]
b = 0.5  # Damping coefficient [N-s/m]

# Nonlinear Pendulum on Cart Dynamics
def pendulum_dynamics(y, t, u):
    # u = Force applied to cart [N]
    x, x_dot, theta, theta_dot = y
    
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    
    # Equations of motion
    x_ddot = (3*L*g*m2*c*s)/(4*L*m1 + 4*L*m2 - 3*L*m2*c**2) - (4*((L*m2*s*theta_dot**2)/2 - u + b*x_dot))/(4*m1 + 4*m2 - 3*m2*c**2)
    theta_ddot = (6*c*((L*m2*s*theta_dot**2)/2 - u + b*x_dot))/(4*L*m1 + 4*L*m2 - 3*L*m2*c**2) - (6*L*g*m2*s*(m1 + m2))/(4*L**2*m2**2 - 3*L**2*m2**2*c**2 + 4*L**2*m1*m2)
    return jnp.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
## Initialize Neural Net
state_dim = 4
control_dim = 1
key, subkey = jax.random.split(jax.random.PRNGKey(int(time.time())))
class SimpleController(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=control_dim)(x)
        return x

# Create Neural Net
model = SimpleController()
dummy_input = jnp.zeros((1, state_dim))
params = model.init(key, dummy_input)['params']

## Load Neural Net Parameters
relative_ckpt_dir = './GPU_flax_checkpoints'
ckpt_dir = os.path.abspath(relative_ckpt_dir)
# Create the CheckpointManager for the *same* directory
mngr_load = ocp.CheckpointManager(directory=ckpt_dir) 
latest_step = mngr_load.latest_step()

if latest_step is not None:
    # Create a target structure to restore into (can be based on params)
    restore_target = {'JAX_LQR_NN16N1HL': params}
    
    # Restore the parameters
    restored_data = mngr_load.restore(latest_step, args=ocp.args.StandardRestore(restore_target))
    loaded_params = restored_data['JAX_LQR_NN16N1HL']
    print(f"Parameters restored from step {latest_step}")
    # Now you can use loaded_params with model.apply
else:
    print("No checkpoint found to restore.")

## Simulate the closed-loop dynamics using odeint
def closed_loop_dynamics(y, t):
    u = jnp.squeeze(model.apply({'params': loaded_params}, y))
    print("Control input u:", u) #debugging
    return pendulum_dynamics(y, t, u)

t_span = (0, 10)
t_eval = jnp.linspace(t_span[0], t_span[1], 101)
y0 = jnp.array([0, 0, jnp.pi + jax.random.uniform(subkey, (), minval=-0.3, maxval=0.3), 0], dtype=float) 
sol_jax = odeint(closed_loop_dynamics, y0, t_eval)

## Plot Simulation Results
time = t_eval
x_pos = sol_jax[:, 0]
theta_angle = sol_jax[:, 2]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x_pos)
plt.xlabel('Time (s)')
plt.ylabel('Cart Position (m)')
plt.title('Cart Position vs. Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, theta_angle)
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle (rad)')
plt.title('Pendulum Angle vs. Time')
plt.grid(True)
plt.tight_layout()
plt.show()