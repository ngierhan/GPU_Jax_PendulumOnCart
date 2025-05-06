import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
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
K = jnp.array(K)

## Simulation
num_simulations = 10
t_span = (0, 10)
t_eval = jnp.linspace(t_span[0], t_span[1], 101)

# Define range for initial theta (around pi)
theta_initial_min = jnp.pi - 0.3 # Increased range slightly
theta_initial_max = jnp.pi + 0.3
initial_thetas = jnp.linspace(theta_initial_min, theta_initial_max, num_simulations)

# Store data from all simulations
all_X_data = []
all_U_data = []

# Simulate the closed-loop dynamics using odeint
def closed_loop_dynamics(y, t):
    u = jnp.squeeze(-jnp.dot(K, (y - OP)))
    return pendulum_dynamics(y, t, u)

print("\nGenerating training data from LQR simulations...")
for i, theta_start in enumerate(initial_thetas):
    y0 = jnp.array([0, 0, theta_start, 0], dtype=float)
    print(f"Running simulation {i+1}/{num_simulations} with initial theta: {theta_start:.4f}")
    
    # Run simulation with LQR controller
    sol_jax = odeint(closed_loop_dynamics, y0, t_eval)
    
    # Store states (X)
    all_X_data.append(sol_jax)
    
    # Calculate and store corresponding LQR control actions (U)
    U_sim = []
    for state in sol_jax:
        state_deviation = state - OP
        u_lqr = jnp.squeeze(-jnp.dot(K, state_deviation))
        # u_lqr = jnp.clip(u_lqr, -20, 20) # Apply same clipping if used in dynamics
        U_sim.append(u_lqr)
    all_U_data.append(jnp.array(U_sim).reshape(-1, 1)) # Ensure U has shape (steps, 1)

# Concatenate data from all simulations
X_train = jnp.concatenate(all_X_data, axis=0)
U_train = jnp.concatenate(all_U_data, axis=0)

print(f"\nTotal training samples generated: {X_train.shape[0]}")


# First step of the simulation results for validation
print("JAX Simulation Results (first few steps):\n", sol_jax[:5])

# Plot Simulation Results
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









## Create and Train Neural Net in JAX (FLAX)
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# Assume you have generated LQR data: states (X) and corresponding LQR control actions (U)
# X should be a jnp.array of shape (num_samples, state_dim)
# U should be a jnp.array of shape (num_samples, control_dim)

# Import LQR data from the simulation
X = X_train
U = U_train
num_samples = X.shape[0]
state_dim = 4
control_dim = 1

# Split training into training and validation sets (80% train, 20% val)
key, subkey = jax.random.split(jax.random.PRNGKey(42))
perm = jax.random.permutation(subkey, num_samples)
X_shuffled_total = X_train[perm]
U_shuffled_total = U_train[perm]

val_fraction = 0.2  # Use 20% for validation
num_val_samples = int(num_samples * val_fraction)
X_val = X_shuffled_total[:num_val_samples]
U_val = U_shuffled_total[:num_val_samples]

X_train_split = X_shuffled_total[num_val_samples:]
U_train_split = U_shuffled_total[num_val_samples:]

# Define the neural network architecture using Flax
class SimpleController(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=control_dim)(x)
        return x

# Initialize the neural network
model = SimpleController()
dummy_input = jnp.zeros((1, state_dim))
params = model.init(key, dummy_input)['params']

# Define the loss function
def loss_fn(params, x_batch, u_batch):
    u_pred = model.apply({'params': params}, x_batch)
    loss = jnp.mean((u_pred - u_batch)**2)  # Mean Squared Error
    return loss

# Define the optimizer
learning_rate = 1e-5
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
# Convergence criteria
patience = 1000  # Number of epochs to wait for improvement before stopping (tune this)
min_delta = 1e-10 # Minimum change in val_loss to qualify as improvement (optional)
epochs_no_improve = 0
best_val_loss = jnp.inf
best_params = None # To store the parameters of the best model

# Define the training step
@jax.jit
def train_step(params, opt_state, x_batch, u_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, u_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# Training loop
num_epochs = 100000000
batch_size = 128
num_batches = num_samples // batch_size

for epoch in range(num_epochs):
    for batch in range(num_batches):
        start_index = batch * batch_size
        end_index = (batch + 1) * batch_size
        x_batch = X_train_split[start_index:end_index]
        u_batch = U_train_split[start_index:end_index]
        params, opt_state, loss = train_step(params, opt_state, x_batch, u_batch)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    val_loss = loss_fn(params, X_val, U_val) 
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_params = params # Store the best parameters found so far
        print(f"  -> New best validation loss: {best_val_loss:.6f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
        print(f"No improvement in validation loss for {patience} consecutive epochs.")
        break 
    # --- End Early Stopping Check ---


# Now 'params' contains the trained weights of the neural network controller
print("\nTrained Neural Network Parameters:\n", jax.tree_util.tree_map(lambda x: x[:2], params))







## Use NN Controller
# Simulation with random initial conditions
t_span = (0, 10)
t_eval = jnp.linspace(t_span[0], t_span[1], 101)

key, subkey = jax.random.split(key)
random_theta_start = jax.random.uniform(subkey, minval=theta_initial_min, maxval=theta_initial_max)
y0 = jnp.array([0, 0, random_theta_start, 0], dtype=float) 
print(f"Using randomized initial condition for NN test: {y0}")

# Simulate the closed-loop dynamics using odeint
def closed_loop_dynamics(y, t):
    u = jnp.squeeze(model.apply({'params': params}, y))
    print("Control input u:", u) #debugging
    return pendulum_dynamics(y, t, u)

print("Initial state y0:", y0, "Shape:", y0.shape) #debugging
print("Time points t_eval:", t_eval, "Shape:", t_eval.shape)
sol_jax = odeint(closed_loop_dynamics, y0, t_eval)

# sol_jax now contains the state trajectory over time
print("JAX Simulation Results (first few steps):\n", sol_jax[:5])

# Plot Simulation Results
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









## Save Neural Network Parameters
import orbax.checkpoint as ocp
import os 

# Decide which parameters to save (prefer best_params if using early stopping)
params_to_save = best_params if best_params is not None else params
final_epoch = epoch + 1 # Or the epoch number where best_params was saved

# Define the directory to save checkpoints
ckpt_dir = './flax_checkpoints' 

# ===> Convert the relative path to an absolute path <===
abs_ckpt_dir = os.path.abspath(ckpt_dir)
print(f"Using absolute checkpoint directory: {abs_ckpt_dir}") # Optional: print to verify

# Ensure the absolute directory exists
os.makedirs(abs_ckpt_dir, exist_ok=True) 

# Create a CheckpointManager using the ABSOLUTE path
mngr = ocp.CheckpointManager(directory=abs_ckpt_dir, # Use abs_ckpt_dir
                             options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))

# Save the parameters
save_data = {'JAX_LQR_NN16N1HL': params_to_save} 
mngr.save(step=final_epoch, args=ocp.args.StandardSave(save_data)) 
mngr.wait_until_finished() # Ensure saving is complete

print(f"Parameters saved to {abs_ckpt_dir} at step {final_epoch}") 






# os.makedirs(ckpt_dir, exist_ok=True) # Ensure directory exists

# # Create a CheckpointManager
# # options can customize behavior, e.g., max_to_keep, create=True
# mngr = ocp.CheckpointManager(directory=ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))

# # Save the parameters
# # We save it within a dictionary structure, which is common practice
# save_data = {'LQR_NN_64N1HL': params_to_save} 
# mngr.save(step=final_epoch, args=ocp.args.StandardSave(save_data)) 
# mngr.wait_until_finished() # Ensure saving is complete

# print(f"Parameters saved to {ckpt_dir} at step {final_epoch}")












# print(sol)
# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(sol.t, sol.y[0], label='Cart position [m]')
# plt.plot(sol.t, sol.y[2], label='Pendulum angle [rad]')
# plt.xlabel('Time (s)')
# plt.ylabel('State values')
# plt.title('Inverted Pendulum on a Cart LQR')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting Input Data
# Fvec = jnp.zeros(sol.y[0].shape)
# print(Fvec)
# print(sol.y[:,0])
# print(float(-jnp.dot(K,(sol.y[:,0]-OP))))
# for i in range(0,sol.nfev):
#     Fvec[i] = float(-jnp.dot(K,(sol.y[:,i]-OP)))

# plt.figure(figsize=(10, 6))
# plt.plot(sol.t, Fvec, label='Force [N]')
# plt.xlabel('Time (s)')
# plt.ylabel('Force [N]')
# plt.title('LQR Input Forces')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(sol.y[0], Fvec, label='Cart position [m]')
# plt.plot(sol.y[2], Fvec, label='Pendulum angle [rad]')
# plt.xlabel('State')
# plt.ylabel('F [N]')
# plt.title('LQR Input Forces')
# plt.legend()
# plt.grid(True)
# plt.show()

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
# fig.suptitle('States vs Input Force')
# ax1.plot(sol.y[0], Fvec)
# ax1.set_title('cart position')
# ax2.plot(sol.y[1], Fvec)
# ax2.set_title('cart velocity')
# ax3.plot(sol.y[2], Fvec)
# ax3.set_title('pendulum angle')
# ax4.plot(sol.y[3], Fvec)
# ax4.set_title('pendulum velocity')
# plt.grid(True)
# plt.show()

# ## Creating and Training Neural Net Controller
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# # Training Data - States as NN inputs and Forces as NN outputs
# inputs = torch.tensor(sol.y.T, dtype=torch.float32)
# outputs = torch.tensor(Fvec, dtype=torch.float32).view(-1, 1) 

# # Create Neural Net 
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork()

# # Hyperparameters
# learning_rate = 1e-3
# num_epochs = 5000

# # Initialize the loss function
# loss_fn = nn.CrossEntropyLoss()

# # Optimizer adjusts the model's parameters to reduce the loss
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # Training
# for epoch in range(num_epochs):
#     # Forward pass
#     predictions = model(inputs)  # Model's predictions
#     loss = loss_fn(predictions, outputs)  # Compute loss

#     # Backward pass
#     optimizer.zero_grad()  # Clear gradients
#     loss.backward()  # Compute gradients
#     optimizer.step()  # Update weights

#     # Print progress
#     if (epoch + 1) % 50 == 0:  # Every 50 epochs
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
              
# # Test the model on the same data
# with torch.no_grad():
#     predicted = model(inputs)
#     print(predicted[:5])  # Print first 5 predictions

# ## Simulation Using Neural Net Controller
# # Closed-loop dynamics with NN control
# def closed_loop_dynamics(t, y):
#     #print(jnp.dot(K,(y-OP)))
#     y_tensor = torch.tensor(y, dtype=torch.float32).view(1, -1)
#     print(y_tensor)
#     u = float(model(y_tensor))
#     dydt = pendulum_dynamics(t, y, u)
#     return jnp.array(dydt, dtype=float)

# t_span = (0, 10)
# t_eval = jnp.linspace(0, 10, 500)
# #y0 = jnp.array([0, 0, jnp.pi + 0.1, 0], dtype=float)  # Initial state as numpy array
# y0 = jnp.array([1, 0, jnp.pi, 0], dtype=float)  # Initial state as numpy array

# # Solve the system using solve_ivp
# sol = solve_ivp(closed_loop_dynamics, t_span, y0, t_eval=t_eval, method='RK45')
# print(sol)
# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(sol.t, sol.y[0], label='Cart position [m]')
# plt.plot(sol.t, sol.y[2], label='Pendulum angle [rad]')
# plt.xlabel('Time (s)')
# plt.ylabel('State values')
# plt.title('Inverted Pendulum on a Cart Neural Net')
# plt.legend()
# plt.grid(True)
# plt.show()