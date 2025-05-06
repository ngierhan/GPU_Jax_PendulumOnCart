import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import optax
from jax import random
import control
import matplotlib.pyplot as plt

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
    
## Use Symbolics to define A and B matrices for LQR controller
from sympy import symbols, Matrix, sin, cos
OP=jnp.array([0,0,jnp.pi,0])
# Define symbolic variables
x_sym, dx_sym, th_sym, dth_sym, F_sym = symbols('x_sym dx_sym th_sym dth_sym F_sym')
m1_sym, b_sym, L_sym, g_sym, m2_sym = symbols('m1_sym b_sym L_sym g_sym m2_sym')
q = Matrix([x_sym, dx_sym, th_sym, dth_sym])  # State variables
# SYMBOLIC Equations of motion
s_sym = sin(th_sym)
c_sym = cos(th_sym)

x_ddot_sym = (3*L_sym*g_sym*m2_sym*c_sym*s_sym)/(4*L_sym*m1_sym + 4*L_sym*m2_sym - 3*L_sym*m2_sym*c_sym**2) - (4*((L_sym*m2_sym*s_sym*dth_sym**2)/2 - F_sym + b*dx_sym))/(4*m1_sym + 4*m2_sym - 3*m2_sym*c_sym**2)
theta_ddot_sym = (6*c_sym*((L_sym*m2_sym*s_sym*dth_sym**2)/2 - F_sym + b_sym*dx_sym))/(4*L_sym*m1_sym + 4*L_sym*m2_sym - 3*L_sym*m2_sym*c_sym**2) - (6*L_sym*g_sym*m2_sym*s_sym*(m1_sym + m2_sym))/(4*L_sym**2*m2_sym**2 - 3*L_sym**2*m2_sym**2*c_sym**2 + 4*L_sym**2*m1_sym*m2_sym)

A_sym = Matrix([dx_sym, x_ddot_sym, dth_sym, theta_ddot_sym]).jacobian(q)
A_OP = A_sym.subs(zip(q,OP))
A = A_OP.subs({m1_sym: m1, b_sym: b, L_sym: L, g_sym: g, m2_sym: m2, F_sym:0})

print("Jacobian Matrix A:")
print(A)     

B_sym = Matrix([dx_sym, x_ddot_sym, dth_sym, theta_ddot_sym]).jacobian(Matrix([F_sym]))
B_OP = B_sym.subs(zip(q,OP))
B = B_OP.subs({m1_sym: m1, b_sym: b, L_sym: L, g_sym: g, m2_sym: m2, F_sym:0})

print("Jacobian Matrix B:")
print(B)  


# LQR controller design
Q = jnp.diag(jnp.array([1, 1, 1, 1])) # State cost matrix
R = jnp.array([[1]], dtype=float) # Control cost matrix

# Calculate the LQR gain matrix
K, S, E = control.lqr(A, B, Q, R)

# Convert K to a JAX array
K_LQR = jnp.array(K)
print("Computed K:\n", K_LQR)






## Determine Gain Matrix using ADAM
# Initialize K
key = random.PRNGKey(0)
initial_K = random.normal(key, (1, 4)) * 0.1 # Start with small gains
params = {'K': initial_K}

# Define the loss function
def calculate_loss(params_local, y0, target, t_eval, control_weight=0.01):
    K_local = params_local['K']

    # Define controller and closed-loop dynamics *inside* the loss function
    # so they capture the K_local being optimized
    def controller(state_deviation):
        # Linear state feedback controller
        u = -jnp.dot(K_local, state_deviation)
        return jnp.squeeze(u) # Ensure u is scalar for pendulum_dynamics

    def closed_loop_dynamics_opt(y, t):
        state_deviation = y - target
        u = controller(state_deviation)
        return pendulum_dynamics(y, t, u)

    # Simulate using odeint
    sol = odeint(closed_loop_dynamics_opt, y0, t_eval, rtol=1e-5, atol=1e-5) # Adjust tolerances if needed

    # Calculate loss components
    state_deviation = sol - target
    state_loss = jnp.mean(state_deviation**2) # Mean squared error from target state

    # Calculate control effort along the trajectory
    # Need to vmap the controller function over the trajectory states
    control_inputs = jax.vmap(controller)(sol - target)
    control_loss = jnp.mean(control_inputs**2)

    # Total weighted loss
    total_loss = state_loss + control_weight * control_loss
    return total_loss

# JIT Compile gradients
value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_loss))

# Define the optimizer
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Train K
num_epochs = 10000
# Define initial state for optimization (can vary this for robustness)
y0_train = jnp.array([0.0, 0.0, jnp.pi + jax.random.uniform(random.PRNGKey(0), (), minval=-0.5, maxval=0.5), 0.0])
t_span_train = (0, 5) # Shorter simulation for faster training steps
t_eval_train = jnp.linspace(t_span_train[0], t_span_train[1], 51) # Fewer points

loss_val_prev = 99999 # Initialize previous loss for learning rate scheduler
for epoch in range(num_epochs):
    # Calculate loss and gradients for the current parameters
    loss_val, grads = value_and_grad_fn(params, y0_train, OP, t_eval_train)

    # Compute updates based on gradients and optimizer state
    updates, opt_state = optimizer.update(grads, opt_state, params)

    # Apply updates to the parameters
    params = optax.apply_updates(params, updates)

    if epoch % 500 == 0 or epoch == num_epochs - 1:
        # LEARNING RATE SCHEDULER
        if loss_val > loss_val_prev:
            learning_rate = learning_rate*.9
            print("Learning rate decreased to:", learning_rate)
        loss_val_prev = loss_val
        print(f"Epoch {epoch}: Loss = {loss_val:.6f} | Learning Rate = {learning_rate:.3e}")

print("Optimization finished.")
optimized_K = params['K']
print("Optimized K:\n", optimized_K)


## Test Optimized K
t_span_eval = (0, 10)
t_eval_eval = jnp.linspace(t_span_eval[0], t_span_eval[1], 201)
y0_eval = jnp.array([0.0, 0.0, jnp.pi + jax.random.uniform(random.PRNGKey(0), (), minval=-0.5, maxval=0.5), 0.0]) 

def final_controller(state_deviation):
     u = -jnp.dot(optimized_K, state_deviation)
     return jnp.squeeze(u)

def final_closed_loop_dynamics(y, t):
    state_deviation = y - OP
    u = final_controller(state_deviation)
    return pendulum_dynamics(y, t, u)

sol_final = odeint(final_closed_loop_dynamics, y0_eval, t_eval_eval)

# Plotting
time = t_eval_eval
x_pos = sol_final[:, 0]
theta_angle = sol_final[:, 2]
control_inputs_final = jax.vmap(final_controller)(sol_final - OP)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, x_pos, label='Optimized K')
plt.axhline(OP[0], color='r', linestyle='--', label='Target x')
plt.xlabel('Time (s)')
plt.ylabel('Cart Position (m)')
plt.title('Optimized Control Simulation')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, theta_angle, label='Optimized K')
plt.axhline(OP[2], color='r', linestyle='--', label='Target theta (pi)')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, control_inputs_final, label='Control Input (Optimized K)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
